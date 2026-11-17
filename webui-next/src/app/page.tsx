import { MapPin, Users, Calendar } from 'lucide-react';
import { operatorApiServerBaseUrl } from '@/lib/operator-api';

export const dynamic = 'force-dynamic';

async function getDashboardData() {
  try {
    const res = await fetch(`${operatorApiServerBaseUrl}/api/v1/dashboard`, { next: { revalidate: 10 } });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function DashboardPage() {
  const data = await getDashboardData();

  // mock fallback if API fails
  const stats = data || {
    activeVenues: 0,
    activeCourts: 0,
    matches7d: 0,
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p className="text-slate-500 mt-2">Overview of Good Badminton activity.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 flex items-center">
          <div className="p-4 bg-indigo-50 rounded-lg text-indigo-600 mr-4">
            <MapPin className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500">Active Venues</p>
            <h3 className="text-2xl font-bold text-slate-900">{stats.activeVenues || stats.active_venues || 0}</h3>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 flex items-center">
          <div className="p-4 bg-emerald-50 rounded-lg text-emerald-600 mr-4">
            <Users className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500">Active Courts</p>
            <h3 className="text-2xl font-bold text-slate-900">{stats.activeCourts || stats.active_courts || 0}</h3>
          </div>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 flex items-center">
          <div className="p-4 bg-blue-50 rounded-lg text-blue-600 mr-4">
            <Calendar className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500">Matches (7d)</p>
            <h3 className="text-2xl font-bold text-slate-900">{stats.matches7d || stats.matches_7d || 0}</h3>
          </div>
        </div>
      </div>
    </div>
  );
}
