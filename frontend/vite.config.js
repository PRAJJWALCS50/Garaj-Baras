import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Vite 8 (rolldown) expects a function here.
        manualChunks(id) {
          if (!id) return
          if (id.includes('node_modules')) {
            if (id.includes('/leaflet/') || id.includes('\\leaflet\\')) return 'leaflet'
            if (id.includes('/react-leaflet/') || id.includes('\\react-leaflet\\'))
              return 'leaflet'
            if (id.includes('/axios/') || id.includes('\\axios\\')) return 'axios'
            return 'vendor'
          }
        },
      },
    },
  },
})
