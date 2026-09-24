'use client';

import { MouseEvent, useMemo, useState } from 'react';
import { apiFetch, CourtOperation, readApiError } from '@/lib/operator-api';

type Point = { x: number; y: number };
type LineKey = 'left' | 'right' | 'cross0' | 'cross1';
type Target = LineKey | 'corner0' | 'corner1' | 'corner2' | 'corner3';
type Mode = 'manual_corners' | 'line_evidence';

const crossLineChoices = [
  { label: '远端底线（0.00m）', value: 0 },
  { label: '远端双打后发球线（0.76m）', value: 0.76 },
  { label: '远端前发球线（4.72m）', value: 4.72 },
  { label: '近端前发球线（8.68m）', value: 8.68 },
  { label: '近端双打后发球线（12.64m）', value: 12.64 },
  { label: '近端底线（13.40m）', value: 13.4 },
];

const targetLabel: Record<Target, string> = {
  left: '左边线（点两处可见部分）',
  right: '右边线（点两处可见部分）',
  cross0: '第一条横线（点两处）',
  cross1: '第二条横线（点两处）',
  corner0: '角 1',
  corner1: '角 2',
  corner2: '角 3',
  corner3: '角 4',
};

function emptyLines(): Record<LineKey, Point[]> {
  return { left: [], right: [], cross0: [], cross1: [] };
}

function isCornerTarget(target: Target): target is 'corner0' | 'corner1' | 'corner2' | 'corner3' {
  return target.startsWith('corner');
}

function selectedPoints(target: Target, lines: Record<LineKey, Point[]>, corners: Point[]): Point[] {
  if (isCornerTarget(target)) return corners[Number(target.slice(-1))] ? [corners[Number(target.slice(-1))]] : [];
  return lines[target];
}

export default function CourtCalibrationPanel({
  venueId,
  operation,
  onSaved,
  onClose,
}: {
  venueId: string;
  operation: CourtOperation;
  onSaved: (message: string) => Promise<void>;
  onClose: () => void;
}) {
  const activeCase = operation.case;
  const [mode, setMode] = useState<Mode>('line_evidence');
  const [target, setTarget] = useState<Target>('left');
  const [lines, setLines] = useState<Record<LineKey, Point[]>>(emptyLines);
  const [corners, setCorners] = useState<Point[]>([]);
  const [crossDistances, setCrossDistances] = useState<[number, number]>([0, 4.72]);
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
  const [candidate, setCandidate] = useState<Point[] | null>(null);
  const [busy, setBusy] = useState<'candidate' | 'save' | null>(null);
  const [error, setError] = useState('');
  const [frameToken] = useState(() => Date.now());

  const payload = useMemo(() => mode === 'manual_corners'
    ? { mode, corners, left_sideline: [], right_sideline: [], cross_lines: [] }
    : {
        mode,
        corners: [],
        left_sideline: lines.left,
        right_sideline: lines.right,
        cross_lines: [
          { court_y_m: crossDistances[0], points: lines.cross0 },
          { court_y_m: crossDistances[1], points: lines.cross1 },
        ],
      }, [corners, crossDistances, lines, mode]);

  if (!activeCase?.preview_url) return null;

  function reset() {
    setLines(emptyLines()); setCorners([]); setCandidate(null); setError(''); setTarget(mode === 'manual_corners' ? 'corner0' : 'left');
  }

  function selectMode(nextMode: Mode) {
    setMode(nextMode); setTarget(nextMode === 'manual_corners' ? 'corner0' : 'left'); setLines(emptyLines()); setCorners([]); setCandidate(null); setError('');
  }

  function addPoint(event: MouseEvent<HTMLDivElement>) {
    if (!frameSize.width || !frameSize.height) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = {
      x: Number((((event.clientX - rect.left) / rect.width) * frameSize.width).toFixed(2)),
      y: Number((((event.clientY - rect.top) / rect.height) * frameSize.height).toFixed(2)),
    };
    setCandidate(null);
    if (isCornerTarget(target)) {
      const index = Number(target.slice(-1));
      setCorners((current) => {
        const next = [...current]; next[index] = point; return next;
      });
      return;
    }
    setLines((current) => ({ ...current, [target]: current[target].length >= 2 ? [point] : [...current[target], point] }));
  }

  async function calculate() {
    setBusy('candidate'); setError('');
    try {
      const response = await apiFetch(`/api/v1/venues/${venueId}/courts/${operation.court.id}/calibration-candidate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const body = await response.json() as { candidate: { court_corners: [number, number][] } };
      setCandidate(body.candidate.court_corners.map(([x, y]) => ({ x, y })));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法计算候选四角。');
    } finally { setBusy(null); }
  }

  async function save() {
    if (!candidate) return;
    setBusy('save'); setError('');
    try {
      const response = await apiFetch(`/api/v1/venues/${venueId}/courts/${operation.court.id}/calibration`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      const body = await response.json() as { message: string };
      await onSaved(body.message); onClose();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '保存场地标定失败。');
    } finally { setBusy(null); }
  }

  const allPoints = mode === 'manual_corners' ? corners : Object.values(lines).flat();
  const activePoints = selectedPoints(target, lines, corners);
  const previewSource = `${activeCase.preview_url}?calibration=${frameToken}`;

  return <section className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-left">
    <div className="flex items-start justify-between gap-3"><div><p className="text-sm font-semibold text-amber-950">标注场地线</p><p className="mt-1 text-xs leading-5 text-amber-900">在冻结的预览帧上标注。近端底线被遮挡时，选择两条边线和任意两条可见横线，系统会外推虚拟四角。</p></div><button type="button" onClick={onClose} className="text-xs text-amber-800 underline">关闭</button></div>
    <div className="mt-3 flex gap-2"><button type="button" onClick={() => selectMode('line_evidence')} className={`rounded-md px-2 py-1 text-xs ${mode === 'line_evidence' ? 'bg-amber-700 text-white' : 'bg-white text-amber-900'}`}>按可见场地线</button><button type="button" onClick={() => selectMode('manual_corners')} className={`rounded-md px-2 py-1 text-xs ${mode === 'manual_corners' ? 'bg-amber-700 text-white' : 'bg-white text-amber-900'}`}>直接点四角</button></div>
    <div className="mt-3 overflow-hidden rounded-lg border border-amber-200 bg-slate-950"><div onClick={addPoint} className="relative cursor-crosshair"><video src={previewSource} onLoadedMetadata={(event) => setFrameSize({ width: event.currentTarget.videoWidth, height: event.currentTarget.videoHeight })} autoPlay muted playsInline controls className="block w-full" />{frameSize.width > 0 && <svg viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} className="pointer-events-none absolute inset-0 h-full w-full">{candidate && <polygon points={candidate.map((point) => `${point.x},${point.y}`).join(' ')} fill="rgba(34,197,94,.12)" stroke="#22c55e" strokeWidth="3" />}{allPoints.map((point, index) => <circle key={`${point.x}-${point.y}-${index}`} cx={point.x} cy={point.y} r="7" fill={activePoints.includes(point) ? '#f97316' : '#facc15'} stroke="#111827" strokeWidth="2" />)}</svg>}</div></div>
    {mode === 'line_evidence' ? <div className="mt-3 space-y-2"><div className="flex flex-wrap gap-2">{(['left', 'right', 'cross0', 'cross1'] as LineKey[]).map((item) => <button type="button" key={item} onClick={() => setTarget(item)} className={`rounded-md px-2 py-1 text-xs ${target === item ? 'bg-slate-900 text-white' : 'bg-white text-slate-700'}`}>{targetLabel[item]}（{lines[item].length}/2）</button>)}</div><div className="grid grid-cols-2 gap-2 text-xs"><label>第一横线<select value={crossDistances[0]} onChange={(event) => setCrossDistances((value) => [Number(event.target.value), value[1]])} className="mt-1 w-full rounded border border-amber-300 bg-white p-1">{crossLineChoices.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label><label>第二横线<select value={crossDistances[1]} onChange={(event) => setCrossDistances((value) => [value[0], Number(event.target.value)])} className="mt-1 w-full rounded border border-amber-300 bg-white p-1">{crossLineChoices.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}</select></label></div></div> : <div className="mt-3 flex flex-wrap gap-2">{(['corner0', 'corner1', 'corner2', 'corner3'] as Target[]).map((item) => <button type="button" key={item} onClick={() => setTarget(item)} className={`rounded-md px-2 py-1 text-xs ${target === item ? 'bg-slate-900 text-white' : 'bg-white text-slate-700'}`}>{targetLabel[item]}（{selectedPoints(item, lines, corners).length}/1）</button>)}</div>}
    <p className="mt-2 text-xs text-amber-900">当前点击：{targetLabel[target]}。同一线第三次点击会从新的第一个点开始。</p>{error && <p role="alert" className="mt-2 text-xs text-rose-700">{error}</p>}<div className="mt-3 flex flex-wrap gap-2"><button type="button" disabled={busy !== null} onClick={() => void calculate()} className="rounded-md bg-slate-900 px-3 py-2 text-xs font-medium text-white disabled:opacity-50">{busy === 'candidate' ? '计算中…' : '计算候选四角'}</button><button type="button" disabled={!candidate || busy !== null} onClick={() => void save()} className="rounded-md bg-emerald-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-50">{busy === 'save' ? '保存中…' : '确认并保存标定'}</button><button type="button" disabled={busy !== null} onClick={reset} className="rounded-md bg-white px-3 py-2 text-xs text-slate-700">重置点位</button></div>{candidate && <p className="mt-2 text-xs leading-5 text-emerald-800">候选四角已叠加为绿色多边形。确认它与可见边线、发球线的透视关系一致后再保存；保存后停止预览并重新开始采集。</p>}</section>;
}
