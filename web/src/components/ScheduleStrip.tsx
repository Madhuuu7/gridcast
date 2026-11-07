/* The 96-block schedule strip.
 *
 * Indian grid scheduling runs on a table of 96 fifteen-minute blocks, and a
 * utility is settled on each of them separately. So this is drawn as 96
 * discrete bars rather than a line with a shaded ribbon: the vertical extent
 * of a bar *is* the prediction interval for that block, which is the thing
 * being declared. A smooth ribbon would imply the forecast is continuous in
 * time, and it is not — it is ninety-six separate commitments.
 *
 * Reading it: bar height is uncertainty, the dark tick is the declared
 * value, a dot is what the grid actually drew, and a vermilion dot is a
 * block the interval missed. Colour across the strip is the cooling cycle,
 * not decoration — the band is widest where the load is hottest.
 */

import { useId, useMemo, useState } from "react";
import { animate, stagger } from "motion";
import { useEffect, useRef } from "react";
import { missed, niceTicks, type BandBlock } from "../lib/bands";
import { blockLabel, mixRamp, mw, rampAt } from "../lib/format";

export type StripBlock = BandBlock;

type Props = {
  blocks: StripBlock[];
  /** Drawn once on first paint; later updates redraw without animating. */
  animateIn?: boolean;
  height?: number;
};

const PAD = { top: 18, right: 10, bottom: 30, left: 54 };

function readToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return value || fallback;
}

export function ScheduleStrip({ blocks, animateIn = true, height = 300 }: Props) {
  const clipId = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [theme, setTheme] = useState(0);

  // The ramp colours live in CSS so the dark scheme can restate them, which
  // means they have to be read back out rather than hard-coded here.
  useEffect(() => {
    const target = document.documentElement;
    const observer = new MutationObserver(() => setTheme((n) => n + 1));
    observer.observe(target, { attributes: true, attributeFilter: ["data-theme"] });
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setTheme((n) => n + 1);
    media.addEventListener("change", onChange);
    return () => {
      observer.disconnect();
      media.removeEventListener("change", onChange);
    };
  }, []);

  const ramp = useMemo(
    () => ({
      cool: readToken("--cool", "#3e6e8e"),
      hot: readToken("--hot", "#e09b2d"),
    }),
    [theme],
  );

  const width = 1000;
  const plotWidth = width - PAD.left - PAD.right;
  const plotHeight = height - PAD.top - PAD.bottom;

  const scale = useMemo(() => {
    const values: number[] = [];
    for (const b of blocks) {
      values.push(b.point);
      if (b.lower !== null && Number.isFinite(b.lower)) values.push(b.lower);
      if (b.upper !== null && Number.isFinite(b.upper)) values.push(b.upper);
      if (b.actual !== null) values.push(b.actual);
    }
    if (!values.length) return { min: 0, max: 1 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const pad = Math.max((max - min) * 0.08, 50);
    return { min: Math.max(0, min - pad), max: max + pad };
  }, [blocks]);

  const y = (value: number) =>
    PAD.top + plotHeight * (1 - (value - scale.min) / (scale.max - scale.min || 1));
  const barWidth = plotWidth / 96;
  const x = (block: number) => PAD.left + block * barWidth;

  // One orchestrated moment: the strip fills left to right as the day is
  // declared. Nothing else on the page animates on its own.
  useEffect(() => {
    if (!animateIn || !svgRef.current) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const bars = Array.from(
      svgRef.current.querySelectorAll<SVGRectElement>("[data-bar]"),
    );
    if (!bars.length) return;
    animate(
      bars,
      { opacity: [0, 1], transform: ["scaleY(0.08)", "scaleY(1)"] },
      { duration: 0.42, delay: stagger(0.004), ease: [0.22, 0.9, 0.3, 1] },
    );
  }, [animateIn, blocks]);

  const ticks = useMemo(() => niceTicks(scale.min, scale.max), [scale]);

  const hovered = hover === null ? null : blocks.find((b) => b.block === hover) ?? null;
  return (
    <div className="strip-frame">
      {/* On a phone the full day compressed to 350px puts six blocks in
          every millimetre, which is a texture rather than a reading. The
          strip keeps a usable minimum width and the day scrolls instead. */}
      <div className="strip-scroll">
      <svg
        ref={svgRef}
        className="strip"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Declared schedule across ${blocks.length} fifteen-minute blocks, with prediction intervals`}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={PAD.left} y={0} width={plotWidth} height={height} />
          </clipPath>
        </defs>

        {ticks.map((value) => (
          <g key={value}>
            <line
              className="grid-rule"
              x1={PAD.left}
              x2={width - PAD.right}
              y1={y(value)}
              y2={y(value)}
            />
            <text x={PAD.left - 9} y={y(value) + 3.5} textAnchor="end">
              {mw(value)}
            </text>
          </g>
        ))}

        <g clipPath={`url(#${clipId})`}>
          {blocks.map((b) => {
            const unbounded = b.lower === null || b.upper === null;
            const top = unbounded ? PAD.top : y(b.upper as number);
            const bottom = unbounded ? PAD.top + plotHeight : y(b.lower as number);
            const colour = mixRamp(rampAt(b.block), ramp.cool, ramp.hot);
            const active = hover === b.block;
            return (
              <g key={b.block}>
                <rect
                  data-bar
                  className="band"
                  x={x(b.block) + 0.6}
                  y={top}
                  width={Math.max(barWidth - 1.2, 1)}
                  height={Math.max(bottom - top, 1.5)}
                  fill={colour}
                  fillOpacity={active ? 0.82 : unbounded ? 0.16 : 0.44}
                  style={{ transformOrigin: `${x(b.block)}px ${bottom}px` }}
                />
                <line
                  className="point-tick"
                  x1={x(b.block) + 0.6}
                  x2={x(b.block) + barWidth - 0.6}
                  y1={y(b.point)}
                  y2={y(b.point)}
                />
                {b.actual !== null && (
                  <circle
                    className={missed(b) ? "miss-dot" : "actual-dot"}
                    cx={x(b.block) + barWidth / 2}
                    cy={y(b.actual)}
                    r={missed(b) ? 2.6 : 1.5}
                  />
                )}
                <rect
                  x={x(b.block)}
                  y={PAD.top}
                  width={barWidth}
                  height={plotHeight}
                  fill="transparent"
                  onMouseEnter={() => setHover(b.block)}
                />
              </g>
            );
          })}
        </g>

        {hovered && (
          <line
            className="hover-rule"
            x1={x(hovered.block) + barWidth / 2}
            x2={x(hovered.block) + barWidth / 2}
            y1={PAD.top}
            y2={PAD.top + plotHeight}
          />
        )}

        <line
          className="axis-rule"
          x1={PAD.left}
          x2={width - PAD.right}
          y1={PAD.top + plotHeight}
          y2={PAD.top + plotHeight}
        />
        {[0, 16, 32, 48, 64, 80, 95].map((block) => (
          <text
            key={block}
            x={x(block) + barWidth / 2}
            y={PAD.top + plotHeight + 16}
            textAnchor="middle"
          >
            {blockLabel(block)}
          </text>
        ))}
      </svg>
      </div>

      <div className="readout" aria-live="polite">
        {hovered ? (
          <>
            <span>
              block <b>{hovered.block}</b> at <b>{blockLabel(hovered.block)}</b>
            </span>
            <span>
              declared <b>{mw(hovered.point)} MW</b>
            </span>
            <span>
              band{" "}
              <b>
                {hovered.lower === null || hovered.upper === null
                  ? "unbounded"
                  : `${mw(hovered.lower)} – ${mw(hovered.upper)} MW`}
              </b>
            </span>
            {hovered.actual !== null && (
              <span className={missed(hovered) ? "missed" : undefined}>
                drew <b>{mw(hovered.actual)} MW</b>
                {missed(hovered) ? " — outside the band" : ""}
              </span>
            )}
          </>
        ) : (
          <span>Point at a block to read its declaration.</span>
        )}
      </div>

      <div className="legend">
        <span>
          <i className="ramp" /> band, shaded by the day&rsquo;s heat
        </span>
        <span>
          <i
            style={{
              height: 2,
              background: "var(--ink)",
              alignSelf: "center",
            }}
          />
          declared value
        </span>
        <span>
          <i
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: "var(--ink)",
            }}
          />
          actual
        </span>
        <span>
          <i
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: "var(--miss)",
            }}
          />
          outside the band
        </span>
      </div>
    </div>
  );
}
