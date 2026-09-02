'use client';

import { FormEvent, useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  CaptureMode, Court, CourtOperation, CourtStatus, GpuExecutionEvent, operatorApiBaseUrl,
  readApiError, Venue, VenueOperationsResponse,
} from '@/lib/operator-api';
import { formatChinaTime } from '@/lib/utils';

const statusLabel: Record<CourtStatus, string> = { active: '可用', maintenance: '维护', inactive: '停用' };

function statePill(ok: boolean, yes: string, no: string) {
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${ok ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>{ok ? yes : no}</span>;
}

function eventText(event: GpuExecutionEvent): string {
  if (event.message) return event.message;
  if (event.payload) return JSON.stringify(event.payload);
  return event.event_type ?? event.type ?? 'GPU 已返回执行事件';
}

export default function CourtManagementClient({ venue, initialCourts }: { venue: Venue; initialCourts: Court[] }) {
  const router = useRouter();
  const [courts, setCourts] = useState(initialCourts);
  const [operations, setOperations] = useState<CourtOperation[]>([]);
  const [events, setEvents] = useState<Record<string, GpuExecutionEvent[]>>({});
  const [name, setName] = useState('');
  const [code, setCode] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [previewTick, setPreviewTick] = useState(0);

  const loadOperations = useCallback(async () => {
    try {
      const response = await fetch(`${operatorApiBaseUrl}/api/v1/venues/${venue.id}/operations`, { cache: 'no-store' });
      if (!response.ok) throw new Error(await readApiError(response));
      const payload = await response.json() as VenueOperationsResponse;
      setOperations(payload.courts);
      const activeCases = payload.courts.map((item) => item.case).filter((item): item is NonNullable<CourtOperation['case']> => Boolean(item?.gpu_analysis_session_id));
      const loaded = await Promise.all(activeCases.map(async (caseItem) => {
        const eventResponse = await fetch(`${operatorApiBaseUrl}/api/v1/cases/${caseItem.id}/gpu-events?limit=8`, { cache: 'no-store' });
        if (!eventResponse.ok) return [caseItem.id, []] as const;
        const eventPayload = await eventResponse.json() as { events?: GpuExecutionEvent[]; persisted_events?: GpuExecutionEvent[] };
        return [caseItem.id, eventPayload.events?.length ? eventPayload.events : (eventPayload.persisted_events ?? [])] as const;
      }));
      setEvents(Object.fromEntries(loaded));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '实时场地状态读取失败。');
    }
  }, [venue.id]);

  useEffect(() => {
    const initialTimer = window.setTimeout(() => void loadOperations(), 0);
    const statusTimer = window.setInterval(() => void loadOperations(), 5000);
    const previewTimer = window.setInterval(() => setPreviewTick((current) => current + 1), 3000);
    return () => { window.clearTimeout(initialTimer); window.clearInterval(statusTimer); window.clearInterval(previewTimer); };
  }, [loadOperations]);

  async function changeStatus(court: Court, nextStatus: CourtStatus) {
    setBusy(court.id); setError('');
    try {
      const response = await fetch(`${operatorApiBaseUrl}/api/v1/venues/${venue.id}/courts/${court.id}/status`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: nextStatus }) });
      if (!response.ok) throw new Error(await readApiError(response));
      const payload = await response.json() as { court: Court };
      setCourts((current) => current.map((item) => item.id === court.id ? payload.court : item));
      setMessage(`${court.name} 已更新为“${statusLabel[nextStatus]}”。`);
      await loadOperations();
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : '场地状态更新失败。'); }
    finally { setBusy(null); }
  }

  async function toggleGpu(operation: CourtOperation) {
    const activeCase = operation.case;
    if (!activeCase || !operation.camera.connected) return;
    setBusy(activeCase.id); setError('');
    try {
      const response = await fetch(`${operatorApiBaseUrl}/api/v1/venues/${venue.id}/courts/${operation.court.id}/case/gpu-forwarding`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: !activeCase.gpu_forwarding_enabled }),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const payload = await response.json() as { message: string };
      setMessage(payload.message); await loadOperations();
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : 'GPU 推送状态更新失败。'); }
    finally { setBusy(null); }
  }

  async function setCaptureMode(operation: CourtOperation, mode: CaptureMode) {
    setBusy(operation.court.id); setError('');
    try {
      const response = await fetch(`${operatorApiBaseUrl}/api/v1/venues/${venue.id}/courts/${operation.court.id}/capture`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const payload = await response.json() as { message: string };
      setMessage(payload.message); await loadOperations();
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : '视频采集状态更新失败。'); }
    finally { setBusy(null); }
  }

  async function addCourt(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy('new'); setError('');
    try {
      const response = await fetch(`${operatorApiBaseUrl}/api/v1/venues/${venue.id}/courts`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, code, sort_order: courts.length, status: 'active' }) });
      if (!response.ok) throw new Error(await readApiError(response));
      const payload = await response.json() as { court: Court };
      setCourts((current) => [...current, payload.court]); setName(''); setCode(''); setMessage('新场地已创建。'); router.refresh(); await loadOperations();
    } catch (requestError) { setError(requestError instanceof Error ? requestError.message : '新场地创建失败。'); }
    finally { setBusy(null); }
  }

  const operationByCourt = new Map(operations.map((item) => [item.court.id, item]));

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-indigo-100 bg-indigo-50 px-5 py-4 text-sm text-indigo-950">
        <strong>实时控制规则：</strong>球馆网关只常驻发送心跳，不需要现场人员操作。未标定时可先开启预览；根据预览保存并验证四角后，停止预览再开始采集。GPU 只处理已标定会话的后续片段。界面时间统一为中国标准时间。
      </div>
      <form onSubmit={addCourt} className="bg-white rounded-xl shadow-sm border border-slate-200 p-5 grid grid-cols-1 md:grid-cols-[1fr_1fr_auto] gap-4 items-end">
        <label className="text-sm text-slate-700">场地名称<input required value={name} onChange={(event) => setName(event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：2 号场" /></label>
        <label className="text-sm text-slate-700">场地编码<input required value={code} onChange={(event) => setCode(event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：court-02" /></label>
        <button disabled={busy !== null} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg font-medium disabled:opacity-50">{busy === 'new' ? '创建中…' : '新增场地'}</button>
      </form>
      {(message || error) && <p role={error ? 'alert' : 'status'} className={`rounded-lg px-4 py-3 text-sm ${error ? 'bg-rose-50 text-rose-700' : 'bg-emerald-50 text-emerald-700'}`}>{error || message}</p>}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-x-auto">
        <table className="min-w-[1320px] w-full text-left border-collapse"><thead><tr className="bg-slate-50 border-b border-slate-200">
          <th className="py-4 px-5 text-sm font-semibold text-slate-600">场地 / 可用性</th><th className="py-4 px-5 text-sm font-semibold text-slate-600">摄像头连接</th><th className="py-4 px-5 text-sm font-semibold text-slate-600">实时视频 / 采集</th><th className="py-4 px-5 text-sm font-semibold text-slate-600">当前 case</th><th className="py-4 px-5 text-sm font-semibold text-slate-600">GPU 推送</th><th className="py-4 px-5 text-sm font-semibold text-slate-600">GPU 状态与输出</th>
        </tr></thead><tbody>{courts.length === 0 ? <tr><td colSpan={6} className="py-8 text-center text-slate-500">该球馆尚未注册场地。</td></tr> : courts.map((court) => {
          const operation = operationByCourt.get(court.id); const activeCase = operation?.case; const caseEvents = activeCase ? events[activeCase.id] ?? [] : [];
          return <tr key={court.id} className="align-top border-b border-slate-100"><td className="py-5 px-5"><p className="font-semibold text-slate-900">{court.name}</p><p className="mt-1 text-xs text-slate-500">{court.code}</p><select aria-label={`${court.name} 状态`} disabled={busy !== null} value={court.status} onChange={(event) => changeStatus(court, event.target.value as CourtStatus)} className="mt-3 border border-slate-300 rounded-lg px-2 py-1.5 text-sm"><option value="active">可用</option><option value="maintenance">维护</option><option value="inactive">停用</option></select></td>
            <td className="py-5 px-5"><div>{statePill(Boolean(operation?.camera.connected), '已连接', '未连接')}</div><p className="mt-2 text-xs text-slate-500">{operation?.camera.camera_code ?? '未绑定摄像头'}</p><p className="mt-1 text-xs text-slate-400">{operation?.camera.camera_heartbeat_at ? `心跳 ${formatChinaTime(operation.camera.camera_heartbeat_at)}` : '等待心跳'}</p></td>
            <td className="py-5 px-5"><div className="w-48"><div>{statePill(operation?.capture.mode !== 'idle', operation?.capture.mode === 'record' ? '采集中' : '预览中', '未采集')}</div>{activeCase?.preview_url && operation?.camera.connected ? <div className="mt-3"><video key={`${activeCase.id}-${previewTick}`} src={`${activeCase.preview_url}?t=${previewTick}`} autoPlay muted playsInline controls className="aspect-video w-full rounded-lg bg-slate-950 object-cover" /><p className="mt-2 text-xs text-slate-500">业务服务器短时预览（约 2–5 秒）</p></div> : <p className="mt-3 text-sm text-slate-500">{operation?.camera.connected ? operation?.capture.mode === 'idle' ? '由后台开启后才上传视频' : '等待首个可播放视频片段' : '摄像头连接后可预览'}</p>}<div className="mt-3 flex flex-wrap gap-2">{operation?.capture.mode === 'idle' ? <><button disabled={!operation?.camera.connected || busy !== null} onClick={() => { if (operation) void setCaptureMode(operation, 'preview'); }} className="rounded-lg bg-slate-100 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50">{busy === court.id ? '请求中…' : '开启预览'}</button><button disabled={!operation?.camera.connected || busy !== null} onClick={() => { if (operation) void setCaptureMode(operation, 'record'); }} className="rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50">开始采集</button></> : <button disabled={busy !== null} onClick={() => { if (operation) void setCaptureMode(operation, 'idle'); }} className="rounded-lg bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700 hover:bg-rose-100 disabled:cursor-not-allowed disabled:opacity-50">{busy === court.id ? '停止中…' : '停止视频'}</button>}</div></div></td>
            <td className="py-5 px-5">{activeCase ? <><code className="block max-w-52 break-all rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">{activeCase.id}</code><p className="mt-2 text-xs text-slate-600">{activeCase.status} · 收到 {activeCase.received_segment_count} 段</p>{activeCase.last_received_at && <p className="mt-1 text-xs text-slate-400">最近接收 {formatChinaTime(activeCase.last_received_at)}</p>}</> : <p className="text-sm text-slate-500">终端尚未创建 case</p>}</td>
            <td className="py-5 px-5">{activeCase ? <>{operation?.camera.calibration_status === 'validated' ? <><div>{statePill(activeCase.gpu_forwarding_enabled, '推送中', '已暂停')}</div><button disabled={!operation?.camera.connected || busy !== null} onClick={() => { if (operation) void toggleGpu(operation); }} className={`mt-3 rounded-lg px-3 py-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${activeCase.gpu_forwarding_enabled ? 'bg-rose-50 text-rose-700 hover:bg-rose-100' : 'bg-indigo-600 text-white hover:bg-indigo-700'}`}>{busy === activeCase.id ? '更新中…' : activeCase.gpu_forwarding_enabled ? '停止推送' : '推送到 GPU'}</button></> : <p className="text-sm text-amber-700">先根据预览保存并验证四角，随后重新开始采集。</p>}<p className="mt-2 text-xs text-slate-500">已转发 {activeCase.forwarded_segment_count} 段</p></> : <p className="text-sm text-slate-500">建立 case 后可控制</p>}</td>
            <td className="py-5 px-5"><div className="max-w-80">{activeCase ? <><p className="text-sm font-medium text-slate-800">{activeCase.gpu_status ?? (activeCase.gpu_forwarding_enabled ? '等待 GPU 回执' : '未推送')}</p>{activeCase.gpu_analysis_session_id && <p className="mt-1 break-all text-xs text-slate-500">GPU: {activeCase.gpu_analysis_session_id}</p>}{activeCase.error && <p className="mt-2 text-xs text-rose-700">{activeCase.error.message}</p>}<div className="mt-3 max-h-28 overflow-auto rounded-lg bg-slate-950 p-2 font-mono text-xs text-slate-100">{caseEvents.length ? caseEvents.slice(0, 4).map((item, index) => <p key={item.event_id ?? `${index}-${eventText(item)}`} className="mb-1 break-words">{item.occurred_at && <span className="text-slate-400">[{formatChinaTime(item.occurred_at)}] </span>}<span className="text-emerald-300">[{item.level ?? 'info'}]</span> {eventText(item)}</p>) : <p className="text-slate-400">尚无 GPU 执行事件</p>}</div></> : <p className="text-sm text-slate-500">暂无 GPU 输出</p>}</div></td>
          </tr>;
        })}</tbody></table>
      </div>
    </div>
  );
}
