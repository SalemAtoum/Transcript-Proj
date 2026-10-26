import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],

  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },

    dedupe: ["react", "react-dom"],
  },

  root: __dirname,

  build: {
    outDir: path.resolve(__dirname, "dist"),
    emptyOutDir: true,
  },

  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
  },

  preview: {
    host: "127.0.0.1",
    port: 4173,
  },
});