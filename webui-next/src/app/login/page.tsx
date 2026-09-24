'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch, readApiError } from '@/lib/operator-api';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setMessage('');
    const response = await apiFetch('/api/v1/auth/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
    setBusy(false);
    if (!response.ok) return setMessage(await readApiError(response));
    router.replace('/'); router.refresh();
  }

  return <main className="grid min-h-screen w-full place-items-center bg-slate-950 p-6"><form onSubmit={submit} className="w-full max-w-md space-y-5 rounded-2xl bg-white p-8 shadow-2xl"><div><p className="text-sm font-semibold text-indigo-600">Good Badminton</p><h1 className="mt-2 text-2xl font-bold">业务平台后台登录</h1></div><label className="block text-sm font-medium">账号<input required autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"/></label><label className="block text-sm font-medium">密码<input required type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2"/></label>{message && <p role="alert" className="text-sm text-rose-600">{message}</p>}<button disabled={busy} className="w-full rounded-lg bg-indigo-600 py-2.5 font-medium text-white disabled:opacity-50">{busy ? '登录中…' : '登录'}</button></form></main>;
}
