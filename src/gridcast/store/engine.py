"""Connecting to the database, and refusing to run against the wrong one.

SQLite locally, Postgres when deployed, one ``DATABASE_URL`` either way.
Three things here are less obvious than they look.

**The scheme has to be rewritten.** Render, Heroku and most managed Postgres
providers hand out ``postgres://``, which SQLAlchemy 2 does not recognise --
it wants ``postgresql+psycopg://``. The failure is an unhelpful
``NoSuchModuleError`` at startup, and the fix belongs here rather than in a
deploy checklist nobody re-reads.

**SQLite does not enforce foreign keys unless asked.** The default is off,
for backwards compatibility with 2004. A cascade that works on Postgres and
silently leaves orphan rows on SQLite is the kind of difference that only
shows up in the one environment you cannot debug, so the pragma goes on every
connection.

**The app refuses to boot when the schema is behind.** Not a warning. The bug
that taught this lesson on an earlier project: ``create_all`` adds missing
*tables* but never missing *columns*, so a database from before a column
existed looks perfectly healthy and fails on the first row that needs it.
Alembic owns the schema and the version is checked at startup.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

#: Where a local run keeps its database when nothing says otherwise. Beside
#: the cached CSVs, because the two are the same kind of thing: local state
#: that is rebuildable and does not belong in the repository.
DEFAULT_SQLITE = "sqlite+pysqlite:///data/gridcast.db"


def database_url(raw: str | None = None) -> str:
    """The connection string, normalised.

    ``postgres://`` and ``postgresql://`` both become ``postgresql+psycopg``
    so the same value works whether it came from a Render dashboard, a Neon
    console or a hand-written ``.env``.
    """
    url = raw if raw is not None else os.environ.get("DATABASE_URL") or DEFAULT_SQLITE
    url = url.strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """An engine with the per-dialect settings each one actually needs."""
    url = database_url(url)
    kwargs: dict[str, object] = {"echo": echo, "future": True}

    if is_sqlite(url):
        # A file path in the URL will not create its parent directory, and
        # the resulting "unable to open database file" names neither the
        # path nor the reason.
        if ":memory:" not in url:
            path = Path(url.split("///", 1)[-1])
            path.parent.mkdir(parents=True, exist_ok=True)
        # The daily job writes while the web process reads. WAL lets those
        # overlap instead of raising "database is locked", and the timeout
        # covers the moment the writer holds the lock.
        kwargs["connect_args"] = {"timeout": 15, "check_same_thread": False}
    else:
        # Free-tier Postgres allows few connections and the instance sleeps.
        # A small pool that recycles before the provider's idle timeout
        # avoids the first-request-after-idle connection error.
        kwargs.update(
            pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=280
        )

    engine = create_engine(url, **kwargs)

    if is_sqlite(url):

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """A transaction that commits on success and rolls back on anything else."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class SchemaOutOfDate(RuntimeError):
    """The database is not at the migration head the code expects."""


def current_revision(engine: Engine) -> str | None:
    """The Alembic revision the database is stamped with, or None."""
    with engine.connect() as connection:
        if not connection.dialect.has_table(connection, "alembic_version"):
            return None
        row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
        return row[0] if row else None


def head_revision() -> str | None:
    """The migration head the installed code defines."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).resolve().parents[3]
    ini = root / "alembic.ini"
    if not ini.exists():
        return None
    config = Config(str(ini))
    config.set_main_option("script_location", str(root / "alembic"))
    return ScriptDirectory.from_config(config).get_current_head()


def require_current_schema(engine: Engine) -> str:
    """Boot-time check: stop rather than serve against a stale schema.

    The alternative is the failure this exists to prevent -- an app that
    starts cleanly, serves most requests, and raises on the first one that
    touches a column the database does not have.
    """
    head = head_revision()
    current = current_revision(engine)
    if head is None:
        return current or "no migrations"
    if current is None:
        raise SchemaOutOfDate(
            "the database has no schema; run `alembic upgrade head` before starting"
        )
    if current != head:
        raise SchemaOutOfDate(
            f"the database is at revision {current} but this code expects {head}; "
            f"run `alembic upgrade head`"
        )
    return current
