/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{vue,js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#e0e9ff',
          500: '#4f6ef7',
          600: '#3a56e8',
          700: '#2d44d1',
          900: '#1a2880',
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
