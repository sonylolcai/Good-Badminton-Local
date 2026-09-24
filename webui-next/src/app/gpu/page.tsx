import GpuClient from './GpuClient';
import { operatorServerFetch } from '@/lib/operator-api-server';

export const dynamic = 'force-dynamic';

async function getGpuStatus() {
  try {
    const res = await operatorServerFetch('/api/v1/gpu/status');
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function GpuServicesPage() {
  const status = await getGpuStatus();

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">GPU Services</h1>
        <p className="text-slate-500 mt-2">Manage your remote AI computation and remote GPUs.</p>
      </div>
      <GpuClient initialStatus={status} />
    </div>
  );
}
