import "./globals.css";
import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "CSV Agent",
  description: "Upload CSVs and chat with a Python-powered agent."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0b1021] text-slate-100">{children}</body>
    </html>
  );
}
