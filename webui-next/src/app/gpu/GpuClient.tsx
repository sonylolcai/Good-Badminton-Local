'use client'

import { useState } from 'react';
import { Server, Activity, Cpu, AlertCircle, RefreshCw } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { operatorApiBaseUrl, readApiError } from '@/lib/operator-api';

type GpuStatus = {
  status?: string;
  message?: string;
  control_hint?: string;
  base_url?: string;
  active_jobs?: number;
  pending_jobs?: number;
  utilization?: string;
} | null;

export default function GpuClient({ initialStatus }: { initialStatus: GpuStatus }) {
  const [status, setStatus] = useState(initialStatus);
  const [baseUrl, setBaseUrl] = useState(initialStatus?.base_url || '');
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const router = useRouter();

  const handleSaveConfig = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch(`${operatorApiBaseUrl}/api/v1/gpu/config`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        base_url: baseUrl,
        api_key: apiKey
      }) });
      if (!res.ok) throw new Error(await readApiError(res));
      setStatus(await res.json());
      setApiKey('');
      router.refresh();
      setMessage('GPU 配置已保存。API Key 不会回显。');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'GPU 配置保存失败。');
    }
    setLoading(false);
  };

  const handleCheckHealth = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${operatorApiBaseUrl}/api/v1/gpu/status`);
      if (!res.ok) throw new Error(await readApiError(res));
      setStatus(await res.json());
      setMessage('GPU 健康状态已刷新。');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'GPU 健康检查失败。');
    }
    setLoading(false);
  };

  const handleOperation = async (op: 'start' | 'stop') => {
    setLoading(true);
    try {
      const res = await fetch(`${operatorApiBaseUrl}/api/v1/gpu/operate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ operation: op }) });
      if (!res.ok) throw new Error(await readApiError(res));
      const result = await res.json();
      setMessage(result.message || `GPU ${op} 操作已提交。`);
      await handleCheckHealth();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `GPU ${op} 操作失败。`);
    }
    setLoading(false);
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
          <Server className="w-5 h-5 mr-2 text-indigo-500" />
          Configure GPU Node
        </h2>
        <form onSubmit={handleSaveConfig} className="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          <div className="md:col-span-5">
            <label className="block text-sm font-medium text-slate-700 mb-1">GPU Base URL</label>
            <input required type="text" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} className="w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="http://192.168.1.100:5000" />
          </div>
          <div className="md:col-span-5">
            <label className="block text-sm font-medium text-slate-700 mb-1">API Key (optional)</label>
            <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} className="w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="Leave blank to keep existing" />
          </div>
          <div className="md:col-span-2">
            <button type="submit" disabled={loading} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-4 rounded-lg font-medium transition-colors disabled:opacity-50 flex justify-center items-center">
              {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : 'Connect'}
            </button>
          </div>
        </form>
      </div>

      {message && <p role="status" className="rounded-xl bg-slate-100 px-4 py-3 text-sm text-slate-700">{message}</p>}

      {!status ? (
        <div className="bg-amber-50 border border-amber-200 text-amber-700 p-6 rounded-xl flex items-center">
          <AlertCircle className="w-6 h-6 mr-3" />
          <div>
            <h4 className="font-semibold">Unable to reach API</h4>
            <p className="text-sm mt-1">Check if your FastAPI server is running.</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
              <Cpu className="w-5 h-5 mr-2 text-blue-500" />
              Compute Status
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">Connection</span>
                <span className="font-medium">
                  {status.status === 'ready' ? <span className="text-emerald-500">已连接</span> : <span className="text-amber-600 text-sm">{status.message || status.control_hint || '待检查'}</span>}
                </span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">Active Jobs</span>
                <span className="font-medium text-slate-900">{status.active_jobs ?? 0}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-100">
                <span className="text-slate-600">Pending Queue</span>
                <span className="font-medium text-slate-900">{status.pending_jobs ?? 0}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-slate-600">GPU Utilization</span>
                <span className="font-medium text-slate-900">{status.utilization ?? '0%'}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-emerald-500" />
              Actions
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Execute control actions against the connected GPU instance.
            </p>
            <div className="space-y-3">
              <button onClick={handleCheckHealth} disabled={loading} className="w-full bg-slate-900 hover:bg-slate-800 text-white py-2 px-4 rounded-lg font-medium transition-colors disabled:opacity-50">
                Refresh Status
              </button>
              <div className="grid grid-cols-2 gap-2">
                <button onClick={() => handleOperation('start')} disabled={loading} className="w-full bg-indigo-50 text-indigo-700 hover:bg-indigo-100 py-2 px-4 rounded-lg font-medium transition-colors border border-indigo-200 disabled:opacity-50">
                  Start Worker
                </button>
                <button onClick={() => handleOperation('stop')} disabled={loading} className="w-full bg-red-50 text-red-700 hover:bg-red-100 py-2 px-4 rounded-lg font-medium transition-colors border border-red-200 disabled:opacity-50">
                  Stop Worker
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
