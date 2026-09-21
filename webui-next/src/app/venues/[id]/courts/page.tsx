import Link from 'next/link';
import CourtManagementClient from './CourtManagementClient';
import { Court, operatorApiBaseUrl, Venue } from '@/lib/operator-api';

async function loadVenue(venueId: string): Promise<{ venue: Venue | null; courts: Court[] }> {
  try {
    const [venueResponse, courtsResponse] = await Promise.all([
      fetch(`${operatorApiBaseUrl}/api/v1/venues/${venueId}`, { cache: 'no-store' }),
      fetch(`${operatorApiBaseUrl}/api/v1/venues/${venueId}/courts`, { cache: 'no-store' }),
    ]);
    if (!venueResponse.ok || !courtsResponse.ok) return { venue: null, courts: [] };
    const venuePayload = await venueResponse.json() as { venue: Venue };
    const courtsPayload = await courtsResponse.json() as { courts: Court[] };
    return { venue: venuePayload.venue, courts: courtsPayload.courts };
  } catch {
    return { venue: null, courts: [] };
  }
}

export default async function CourtsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const { venue, courts } = await loadVenue(id);

  if (!venue) {
    return <div className="p-8"><Link href="/venues" className="text-indigo-600 hover:text-indigo-800 text-sm font-medium">← 返回球馆</Link><h1 className="text-2xl font-bold text-slate-900 mt-6">未找到球馆</h1><p className="text-slate-500 mt-2">请确认链接和运营 API 服务是否正确。</p></div>;
  }

  return (
    <div className="p-8">
      <div className="mb-6"><Link href="/venues" className="text-indigo-600 hover:text-indigo-800 text-sm font-medium inline-block">← 返回球馆</Link><h1 className="text-3xl font-bold text-slate-900 mt-4">{venue.name} · 场地管理</h1><p className="text-slate-500 mt-2">{venue.code} · 修改场地可用性会立即影响后续终端会话的可用判断。</p></div>
      <CourtManagementClient venue={venue} initialCourts={courts} />
    </div>
  );
}
