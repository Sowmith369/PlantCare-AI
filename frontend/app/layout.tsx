import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PlantCare AI — Understand your plant",
  description: "AI-assisted plant leaf disease screening and plant health guidance.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
