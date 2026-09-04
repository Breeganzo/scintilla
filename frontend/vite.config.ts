import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The dev server proxies /api to Django rather than calling it cross-origin.
// Two reasons, and the second matters more:
//
//   1. No CORS preflight in development, so the frontend never depends on a
//      permissive CORS setting that would then be tempting to ship.
//   2. The browser sees a same-origin request, which is what it will see in
//      production behind Cloudflare too. Developing against a different origin
//      shape than you deploy is how cookie and header bugs survive until
//      launch day.
//
// Port 8001, not 8000: 8000 has a stale runserver on this machine.
const DJANGO_ORIGIN = process.env.VITE_API_ORIGIN ?? "http://127.0.0.1:8001";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: DJANGO_ORIGIN,
        changeOrigin: true,
      },
    },
  },
  build: {
    // Fail the build rather than shipping a bundle whose size nobody noticed.
    chunkSizeWarningLimit: 600,
    sourcemap: true,
  },
});
