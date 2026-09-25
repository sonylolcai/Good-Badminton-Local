'use client';

import { useCallback, useEffect, useState } from 'react';
import { apiFetch, AdminPrincipal, isPlatformAdmin, readApiError, Venue } from '@/lib/operator-api';

type Player = { id: string; nickname: string };
type MediaResource = { id: string; venue_id: string; player_id: string | null; original_filename: string; upload_succeeded_at: string; status: string; size_bytes: number };

const defaultCorners = JSON.stringify([{ x: 0, y: 0 }, { x: 1920, y: 0 }, { x: 1920, y: 1080 }, { x: 0, y: 1080 }], null, 2);

export default function ResourcesPage() {
  const [platform, setPlatform] = useState(false);
  const [venues, setVenues] = useState<Venue[]>([]);
  const [venueId, setVenueId] = useState('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [resources, setResources] = useState<MediaResource[]>([]);
  const [video, setVideo] = useState<File | null>(null);
  const [playerId, setPlayerId] = useState('');
  const [analysisAssetId, setAnalysisAssetId] = useState('');
  const [template, setTemplate] = useState<File | null>(null);
  const [corners, setCorners] = useState(defaultCorners);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  const loadVenueData = useCallback(async (selected: string) => {
    if (!selected) return;
    const [resourcesResponse, playersResponse] = await Promise.all([apiFetch(`/api/v1/resources?venue_id=${encodeURIComponent(selected)}`), apiFetch('/api/v1/players')]);
    if (resourcesResponse.ok) setResources(((await resourcesResponse.json()) as { resources: MediaResource[] }).resources);
    if (playersResponse.ok) setPlayers(((await playersResponse.json()) as { players: Player[] }).players);
  }, []);

  useEffect(() => { void (async () => {
    const [meResponse, venuesResponse] = await Promise.all([apiFetch('/api/v1/auth/me'), apiFetch('/api/v1/venues')]);
    if (!meResponse.ok || !venuesResponse.ok) return;
    setPlatform(isPlatformAdmin(((await meResponse.json()) as { admin: AdminPrincipal }).admin));
    const nextVenues = ((await venuesResponse.json()) as { venues: Venue[] }).venues;
    setVenues(nextVenues); const first = nextVenues[0]?.id ?? ''; setVenueId(first); await loadVenueData(first);
  })(); }, [loadVenueData]);

  async function upload(event: React.FormEvent) {
    event.preventDefault(); if (!video) return;
    setBusy(true); const form = new FormData(); form.set('video', video); form.set('venue_id', venueId); if (playerId) form.set('player_id', playerId);
    const response = await apiFetch('/api/v1/resources', { method: 'POST', body: form }); setBusy(false);
    if (!response.ok) return setMessage(await readApiError(response));
    setVideo(null); setMessage('视频上传成功，已记录上传成功时间。'); await loadVenueData(venueId);
  }

  async function remove(resource: MediaResource, full: boolean) {
    const warning = full ? '将删除视频、解析数据和整条记录，且无法恢复。确认继续？' : '将删除本地和远程视频，保留记录与 JSON 数据。确认继续？';
    if (!window.confirm(warning)) return;
    setBusy(true); const response = await apiFetch(`/api/v1/resources/${resource.id}${full ? '' : '/resources'}`, { method: 'DELETE' }); setBusy(false);
    if (!response.ok) return setMessage(await readApiError(response));
    setMessage(full ? '全部数据已删除。' : '视频资源已删除，记录与解析数据已保留。'); await loadVenueData(venueId);
  }

  async function analyze(event: React.FormEvent) {
    event.preventDefault(); if (!template || !analysisAssetId) return;
    try { JSON.parse(corners); } catch { return setMessage('球场四角坐标必须是有效 JSON。'); }
    setBusy(true); const form = new FormData(); form.set('template', template); form.set('corners_json', corners);
    const response = await apiFetch(`/api/v1/resources/${analysisAssetId}/analysis`, { method: 'POST', body: form }); setBusy(false);
    if (!response.ok) return setMessage(await readApiError(response));
    setMessage('解析任务已提交。');
  }

  return <div className="space-y-8 p-8"><header><h1 className="text-3xl font-bold">视频资源</h1><p className="mt-2 text-slate-500">管理业务平台上传的视频、关联球员并手动触发解析。</p></header>{message && <p role="status" className="rounded-lg bg-indigo-50 px-4 py-3 text-sm text-indigo-700">{message}</p>}
    <section className="rounded-xl border border-slate-200 bg-white p-6"><div className="flex flex-wrap items-center justify-between gap-4"><h2 className="text-lg font-semibold">资源列表</h2><label className="text-sm">场馆<select value={venueId} onChange={(event) => { setVenueId(event.target.value); setPlayerId(''); void loadVenueData(event.target.value); }} className="ml-2 rounded-lg border border-slate-300 px-3 py-2">{venues.map((venue) => <option key={venue.id} value={venue.id}>{venue.name}</option>)}</select></label></div><form onSubmit={upload} className="mt-5 flex flex-wrap items-end gap-3"><label className="text-sm font-medium">视频<input required accept="video/mp4,video/quicktime,video/x-matroska,video/x-msvideo,video/webm" type="file" onChange={(event) => setVideo(event.target.files?.[0] ?? null)} className="mt-1 block text-sm"/></label><label className="text-sm font-medium">关联球员（可选）<select value={playerId} onChange={(event) => setPlayerId(event.target.value)} className="mt-1 block rounded-lg border border-slate-300 px-3 py-2"><option value="">不关联</option>{players.map((player) => <option key={player.id} value={player.id}>{player.nickname}</option>)}</select></label><button disabled={busy || !venueId} className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white disabled:opacity-50">上传视频</button></form><div className="mt-6 overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="border-b text-slate-500"><th className="py-3">文件</th><th>上传成功时间</th><th>大小</th><th>状态</th><th>操作</th></tr></thead><tbody>{resources.map((resource) => <tr key={resource.id} className="border-b border-slate-100"><td className="py-4 font-medium">{resource.original_filename || resource.id}</td><td>{new Date(resource.upload_succeeded_at).toLocaleString('zh-CN')}</td><td>{(Number(resource.size_bytes) / 1024 / 1024).toFixed(1)} MB</td><td>{resource.status === 'resources_deleted' ? '仅保留数据' : resource.status === 'active' ? '视频可用' : resource.status}</td><td><div className="flex flex-wrap gap-2"><button disabled={resource.status === 'resources_deleted' || busy} onClick={() => { setAnalysisAssetId(resource.id); setMessage('请在下方选择场地图并确认四角坐标。'); }} className="text-indigo-600 disabled:text-slate-300">触发解析</button><button disabled={resource.status === 'resources_deleted' || busy} onClick={() => void remove(resource, false)} className="text-amber-700 disabled:text-slate-300">仅删除资源</button>{platform && <button disabled={busy} onClick={() => void remove(resource, true)} className="text-rose-700">删除全部数据</button>}</div></td></tr>)}{!resources.length && <tr><td colSpan={5} className="py-8 text-center text-slate-500">暂无视频资源。</td></tr>}</tbody></table></div></section>
    {analysisAssetId && <section className="rounded-xl border border-indigo-200 bg-indigo-50 p-6"><h2 className="text-lg font-semibold">手动触发解析</h2><p className="mt-1 text-sm text-slate-600">上传一张球场参考图，并填写视频画面中四个球场角点坐标。</p><form onSubmit={analyze} className="mt-4 grid gap-4 md:grid-cols-2"><label className="text-sm font-medium">球场参考图<input required accept="image/png,image/jpeg,image/bmp" type="file" onChange={(event) => setTemplate(event.target.files?.[0] ?? null)} className="mt-2 block"/></label><label className="text-sm font-medium">四角坐标 JSON<textarea required rows={8} value={corners} onChange={(event) => setCorners(event.target.value)} className="mt-2 block w-full rounded-lg border border-slate-300 p-3 font-mono text-xs"/></label><div className="flex gap-2"><button disabled={busy} className="rounded-lg bg-indigo-600 px-5 py-2 font-medium text-white disabled:opacity-50">提交解析</button><button type="button" onClick={() => setAnalysisAssetId('')} className="rounded-lg border border-slate-300 px-5 py-2">取消</button></div></form></section>}
  </div>;
}
