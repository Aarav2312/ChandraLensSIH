import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The frontend uses same-origin paths (/api, /datasets) so the production build
// works when FastAPI serves dist/. In dev, proxy those to the backend.
const backend = process.env.BACKEND_URL || 'http://localhost:8080';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    host: true,
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/datasets': { target: backend, changeOrigin: true }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false
  }
});
