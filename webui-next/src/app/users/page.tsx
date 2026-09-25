'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiFetch, AdminPrincipal, isPlatformAdmin, readApiError, Venue } from '@/lib/operator-api';

type AdminRow = { id: string; username: string; status: string; must_change_password: boolean; role: string; venue_id: string | null };
type Player = { id: string; nickname: string; status: string; venue_id: string; venue_name: string };

export default function UsersPage() {
  const [me, setMe] = useState<AdminPrincipal | null>(null);
  const [venues, setVenues] = useState<Venue[]>([]);
  const [venueId, setVenueId] = useState('');
  const [admins, setAdmins] = useState<AdminRow[]>([]);
  const [players, setPlayers] = useState<Player[]>([]);
  const [message, setMessage] = useState('');
  const [playerName, setPlayerName] = useState('');
  const [adminForm, setAdminForm] = useState({ username: '', password: '', role: 'venue_admin', venue_id: '' });

  const loadPlayers = useCallback(async (selected: string) => {
    if (!selected) return setPlayers([]);
    const response = await apiFetch(`/api/v1/venues/${selected}/players`);
    if (response.ok) setPlayers(((await response.json()) as { players: Player[] }).players);
  }, []);

  useEffect(() => { void (async () => {
    const [meResponse, venuesResponse] = await Promise.all([apiFetch('/api/v1/auth/me'), apiFetch('/api/v1/venues')]);
    if (!meResponse.ok || !venuesResponse.ok) return;
    const principal = ((await meResponse.json()) as { admin: AdminPrincipal }).admin;
    const nextVenues = ((await venuesResponse.json()) as { venues: Venue[] }).venues;
    setMe(principal); setVenues(nextVenues);
    const firstVenue = nextVenues[0]?.id ?? '';
    setVenueId(firstVenue); setAdminForm((current) => ({ ...current, venue_id: firstVenue }));
    await loadPlayers(firstVenue);
    if (isPlatformAdmin(principal)) {
      const response = await apiFetch('/api/v1/admins');
      if (response.ok) setAdmins(((await response.json()) as { admins: AdminRow[] }).admins);
    }
  })(); }, [loadPlayers]);

  async function createPlayer(event: React.FormEvent) {
    event.preventDefault();
    const response = await apiFetch(`/api/v1/venues/${venueId}/players`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nickname: playerName }) });
    if (!response.ok) return setMessage(await readApiError(response));
    setPlayerName(''); setMessage('球员资产已创建。'); await loadPlayers(venueId);
  }

  async function createAdmin(event: React.FormEvent) {
    event.preventDefault();
    const payload = { ...adminForm, venue_id: adminForm.role === 'platform_admin' ? null : adminForm.venue_id };
    const response = await apiFetch('/api/v1/admins', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!response.ok) return setMessage(await readApiError(response));
    setMessage('管理员已创建，首次登录必须修改密码。'); setAdminForm({ username: '', password: '', role: 'venue_admin', venue_id: venueId });
    const refreshed = await apiFetch('/api/v1/admins');
    if (refreshed.ok) setAdmins(((await refreshed.json()) as { admins: AdminRow[] }).admins);
  }

  async function setAdminStatus(admin: AdminRow) {
    const status = admin.status === 'active' ? 'disabled' : 'active';
    if (status === 'disabled' && !window.confirm(`确认停用管理员 ${admin.username}？`)) return;
    const response = await apiFetch(`/api/v1/admins/${admin.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) });
    if (!response.ok) return setMessage(await readApiError(response));
    setMessage(status === 'active' ? '管理员已启用。' : '管理员已停用，会话已撤销。');
    const refreshed = await apiFetch('/api/v1/admins');
    if (refreshed.ok) setAdmins(((await refreshed.json()) as { admins: AdminRow[] }).admins);
  }

  async function editPlayer(player: Player, status = player.status) {
    const nickname = window.prompt('球员昵称', player.nickname);
    if (nickname === null || !nickname.trim()) return;
    const response = await apiFetch(`/api/v1/venues/${venueId}/players/${player.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nickname, status }) });
    if (!response.ok) return setMessage(await readApiError(response));
    setMessage('球员资产已更新。'); await loadPlayers(venueId);
  }

  const platform = Boolean(me && isPlatformAdmin(me));
  return <div className="space-y-8 p-8"><header><h1 className="text-3xl font-bold">用户与球员</h1><p className="mt-2 text-slate-500">管理员可以登录后台；球员仅作为场馆业务资产。</p></header>{message && <p role="status" className="rounded-lg bg-indigo-50 px-4 py-3 text-sm text-indigo-700">{message}</p>}
    {platform && <section className="rounded-xl border border-slate-200 bg-white p-6"><h2 className="text-lg font-semibold">管理员账号</h2><form onSubmit={createAdmin} className="mt-4 grid gap-3 md:grid-cols-5"><input required minLength={3} value={adminForm.username} onChange={(event) => setAdminForm({ ...adminForm, username: event.target.value })} placeholder="登录账号" className="rounded-lg border border-slate-300 px-3 py-2"/><input required minLength={12} type="password" value={adminForm.password} onChange={(event) => setAdminForm({ ...adminForm, password: event.target.value })} placeholder="临时密码（至少12位）" className="rounded-lg border border-slate-300 px-3 py-2"/><select value={adminForm.role} onChange={(event) => setAdminForm({ ...adminForm, role: event.target.value })} className="rounded-lg border border-slate-300 px-3 py-2"><option value="venue_admin">球馆管理员</option><option value="platform_admin">平台管理员</option></select><select disabled={adminForm.role === 'platform_admin'} value={adminForm.venue_id} onChange={(event) => setAdminForm({ ...adminForm, venue_id: event.target.value })} className="rounded-lg border border-slate-300 px-3 py-2 disabled:bg-slate-100">{venues.map((venue) => <option key={venue.id} value={venue.id}>{venue.name}</option>)}</select><button className="rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white">创建管理员</button></form><div className="mt-5 overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b text-slate-500"><th className="py-2">账号</th><th>角色</th><th>场馆</th><th>状态</th><th>操作</th></tr></thead><tbody>{admins.map((admin) => <tr key={`${admin.id}-${admin.role}-${admin.venue_id}`} className="border-b border-slate-100"><td className="py-3">{admin.username}</td><td>{admin.role === 'platform_admin' ? '平台管理员' : '球馆管理员'}</td><td>{venues.find((venue) => venue.id === admin.venue_id)?.name ?? '全部'}</td><td>{admin.status === 'active' ? '可用' : '已停用'}</td><td><button type="button" onClick={() => void setAdminStatus(admin)} className="text-indigo-600">{admin.status === 'active' ? '停用' : '启用'}</button></td></tr>)}</tbody></table></div></section>}
    <section className="rounded-xl border border-slate-200 bg-white p-6"><div className="flex flex-wrap items-end justify-between gap-4"><div><h2 className="text-lg font-semibold">球员资产</h2><label className="mt-3 block text-sm text-slate-600">场馆<select value={venueId} onChange={(event) => { setVenueId(event.target.value); void loadPlayers(event.target.value); }} className="ml-2 rounded-lg border border-slate-300 px-3 py-2">{venues.map((venue) => <option key={venue.id} value={venue.id}>{venue.name}</option>)}</select></label></div><form onSubmit={createPlayer} className="flex gap-2"><input required value={playerName} onChange={(event) => setPlayerName(event.target.value)} placeholder="球员昵称" className="rounded-lg border border-slate-300 px-3 py-2"/><button disabled={!venueId} className="rounded-lg bg-slate-900 px-4 py-2 font-medium text-white disabled:opacity-50">新增球员</button></form></div><div className="mt-5 grid gap-3 md:grid-cols-3">{players.length ? players.map((player) => <article key={player.id} className="rounded-lg border border-slate-200 p-4"><p className="font-medium">{player.nickname}</p><p className="mt-1 text-xs text-slate-500">{player.status === 'active' ? '可用' : '停用'} · 无登录权限</p><div className="mt-3 flex gap-3 text-sm"><button type="button" onClick={() => void editPlayer(player)} className="text-indigo-600">编辑</button><button type="button" onClick={() => void editPlayer(player, player.status === 'active' ? 'disabled' : 'active')} className="text-slate-600">{player.status === 'active' ? '停用' : '启用'}</button></div></article>) : <p className="text-sm text-slate-500">该场馆暂无球员。</p>}</div></section>
  </div>;
}
