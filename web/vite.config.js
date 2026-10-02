import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Proxy API vers le backend FastAPI (studio_server.py) en développement.
// En production, servez `dist/` derrière le même domaine que l'API, ou définissez
// VITE_API_BASE au moment du build.
const API_TARGET = process.env.VITE_API_TARGET || 'http://127.0.0.1:7860';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': { target: API_TARGET, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
