import { defineConfig, loadEnv } from 'vite';
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const proxy = { target: env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:7863', changeOrigin: false };
  return { base: '/', build: { outDir: 'dist', emptyOutDir: true }, server: { proxy: { '/v1': proxy, '/api': proxy } } };
});
