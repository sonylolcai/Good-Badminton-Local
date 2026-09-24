'use client';

import { useEffect, useState } from 'react';
import { apiFetch, readApiError } from '@/lib/operator-api';

type Policy = { enabled: boolean; retention_days: number; timezone: string; daily_run_time: string; last_started_at?: string | null; last_completed_at?: string | null };

export default function SettingsPage() {
  const [policy, setPolicy] = useState<Policy | null>(null);
  const [message, setMessage] = useState('');
  useEffect(() => { apiFetch('/api/v1/settings/video-retention').then(async (response) => response.ok ? setPolicy((await response.json() as { policy: Policy }).policy) : setMessage(await readApiError(response))); }, []);

  async function save(event: React.FormEvent) {
    event.preventDefault();
    const response = await apiFetch('/api/v1/settings/video-retention', { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(policy) });
    if (!response.ok) return setMessage(await readApiError(response));
    setPolicy((await response.json() as { policy: Policy }).policy); setMessage('视频保留策略已保存。');
  }

  return <div className="p-8"><header><h1 className="text-3xl font-bold">系统设置</h1><p className="mt-2 text-slate-500">由平台管理员统一配置所有平台的视频清理规则。</p></header><section className="mt-8 max-w-2xl rounded-xl border border-slate-200 bg-white p-6"><h2 className="text-lg font-semibold">视频保留策略</h2><p className="mt-2 text-sm text-slate-500">开启后每天清理超过保留期的视频资源，JSON 解析数据与记录不会自动删除。</p>{policy ? <form onSubmit={save} className="mt-5 space-y-4"><label className="flex items-center gap-3"><input type="checkbox" checked={policy.enabled} onChange={(event) => setPolicy({ ...policy, enabled: event.target.checked })} className="h-5 w-5"/><span className="font-medium">启用每日自动清理</span></label><label className="block text-sm font-medium">保留天数<input required min={1} max={3650} type="number" value={policy.retention_days} onChange={(event) => setPolicy({ ...policy, retention_days: Number(event.target.value) })} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2"/></label><label className="block text-sm font-medium">每日执行时间<input required type="time" value={policy.daily_run_time.slice(0, 5)} onChange={(event) => setPolicy({ ...policy, daily_run_time: event.target.value })} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2"/></label><label className="block text-sm font-medium">时区<input required value={policy.timezone} onChange={(event) => setPolicy({ ...policy, timezone: event.target.value })} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2"/></label><button className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white">保存设置</button></form> : <p className="mt-5 text-sm text-slate-500">正在读取设置…</p>}{message && <p role="status" className="mt-4 text-sm text-indigo-700">{message}</p>}</section></div>;
}
