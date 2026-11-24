import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        forest: { 950: "#10291f", 900: "#173b2c", 800: "#20513a", 700: "#286747" },
        leaf: { 50: "#f2f8ef", 100: "#e6f1df", 200: "#cfe4c2", 500: "#6f9f56", 600: "#568743" },
        cream: "#fbfcf8",
      },
      boxShadow: {
        soft: "0 18px 55px rgba(28, 61, 41, 0.08)",
      },
    },
  },
  plugins: [],
};
export default config;
