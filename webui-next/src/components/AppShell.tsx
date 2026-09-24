'use client';

import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { apiFetch, AdminPrincipal, isPlatformAdmin, readApiError } from '@/lib/operator-api';
import { Sidebar } from './Sidebar';

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [admin, setAdmin] = useState<AdminPrincipal | null>();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (pathname === '/login') return;
    apiFetch('/api/v1/auth/me')
      .then(async (response) => {
        if (!response.ok) throw new Error(await readApiError(response));
        const payload = await response.json() as { admin: AdminPrincipal };
        setAdmin(payload.admin);
      })
      .catch(() => router.replace('/login'));
  }, [pathname, router]);

  if (pathname === '/login') return <>{children}</>;
  if (!admin) return <div className="grid min-h-screen flex-1 place-items-center text-slate-500">正在验证登录状态…</div>;

  async function changePassword(event: React.FormEvent) {
    event.preventDefault();
    const response = await apiFetch('/api/v1/auth/change-password', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    });
    if (!response.ok) return setMessage(await readApiError(response));
    setAdmin({ ...admin!, must_change_password: false });
  }

  async function logout() {
    await apiFetch('/api/v1/auth/logout', { method: 'POST' });
    router.replace('/login');
  }

  if (admin.must_change_password) {
    return <main className="grid min-h-screen w-full place-items-center bg-slate-100 p-6"><form onSubmit={changePassword} className="w-full max-w-md space-y-4 rounded-2xl bg-white p-8 shadow-sm"><h1 className="text-2xl font-bold">首次登录请修改密码</h1><p className="text-sm text-slate-500">新密码至少 12 位。</p><input required type="password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} placeholder="当前密码" className="w-full rounded-lg border border-slate-300 px-3 py-2"/><input required minLength={12} type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} placeholder="新密码" className="w-full rounded-lg border border-slate-300 px-3 py-2"/>{message && <p role="alert" className="text-sm text-rose-600">{message}</p>}<button className="w-full rounded-lg bg-indigo-600 py-2 font-medium text-white">保存新密码</button></form></main>;
  }

  return <><Sidebar platformAdmin={isPlatformAdmin(admin)} username={admin.username} onLogout={logout}/><main className="flex-1 overflow-y-auto">{children}</main></>;
}
