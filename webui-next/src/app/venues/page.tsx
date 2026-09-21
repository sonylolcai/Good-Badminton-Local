import Link from 'next/link';
import AddVenueModal from './AddVenueModal';
import { operatorApiBaseUrl, Tenant, Venue } from '@/lib/operator-api';

async function loadData(): Promise<{ venues: Venue[]; tenants: Tenant[] }> {
  try {
    const [venuesResponse, tenantsResponse] = await Promise.all([
      fetch(`${operatorApiBaseUrl}/api/v1/venues`, { cache: 'no-store' }),
      fetch(`${operatorApiBaseUrl}/api/v1/tenants`, { cache: 'no-store' }),
    ]);
    const venuesPayload = venuesResponse.ok ? await venuesResponse.json() as { venues?: Venue[] } : {};
    const tenantsPayload = tenantsResponse.ok ? await tenantsResponse.json() as { tenants?: Tenant[] } : {};
    return { venues: venuesPayload.venues ?? [], tenants: tenantsPayload.tenants ?? [] };
  } catch {
    return { venues: [], tenants: [] };
  }
}

const venueStatusLabel = { active: '可用', inactive: '停用' } as const;

export default async function VenuesPage() {
  const { venues, tenants } = await loadData();

  return (
    <div className="p-8">
      <div className="mb-8 flex justify-between items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">球馆与场地</h1>
          <p className="text-slate-500 mt-2">注册后即可管理场地状态，并为现场终端绑定摄像头。</p>
        </div>
        <AddVenueModal tenants={tenants} />
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead><tr className="bg-slate-50 border-b border-slate-200"><th className="py-4 px-6 text-sm font-semibold text-slate-600">球馆</th><th className="py-4 px-6 text-sm font-semibold text-slate-600">地址</th><th className="py-4 px-6 text-sm font-semibold text-slate-600">场地数</th><th className="py-4 px-6 text-sm font-semibold text-slate-600">状态</th><th className="py-4 px-6 text-sm font-semibold text-slate-600">操作</th></tr></thead>
          <tbody>
            {venues.length === 0 ? <tr><td colSpan={5} className="py-8 text-center text-slate-500">尚未注册球馆。请先创建租户、球馆和首批场地。</td></tr> : venues.map((venue) => (
              <tr key={venue.id} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                <td className="py-4 px-6"><span className="font-medium text-slate-900">{venue.name}</span><div className="text-xs text-slate-400 mt-1">{venue.code}</div></td>
                <td className="py-4 px-6 text-slate-600">{venue.address ?? '未填写'}</td>
                <td className="py-4 px-6 text-slate-600">{venue.court_count}</td>
                <td className="py-4 px-6"><span className={`px-3 py-1 rounded-full text-xs font-medium ${venue.status === 'active' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-700'}`}>{venueStatusLabel[venue.status]}</span></td>
                <td className="py-4 px-6"><Link href={`/venues/${venue.id}/courts`} className="text-indigo-600 hover:text-indigo-800 font-medium text-sm">管理场地 →</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
