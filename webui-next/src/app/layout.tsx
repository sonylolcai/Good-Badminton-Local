import type { Metadata } from 'next';
import './globals.css';
import { AppShell } from '@/components/AppShell';

export const metadata: Metadata = {
  title: 'Good Badminton SaaS',
  description: 'Admin dashboard for Good Badminton',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="flex h-screen overflow-hidden bg-slate-50 font-sans text-slate-900">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
