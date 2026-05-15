export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        intel: {
          blue: '#0071c5',
          'blue-dark': '#005a9e',
          dark: '#2B2C30',
          gray: '#6A6D75',
        },
      },
      fontFamily: {
        display: ['"IntelOne Display"', '"Inter"', 'system-ui', 'sans-serif'],
        text: ['"IntelOne Text"', '"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"Roboto Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
