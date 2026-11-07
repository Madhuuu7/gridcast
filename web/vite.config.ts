import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI service serves the built console from web/dist and mounts its
// assets at /assets, so the build goes there and the dev server proxies the
// API rather than duplicating it.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
