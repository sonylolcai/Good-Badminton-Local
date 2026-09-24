'use client';

import { FormEvent, useMemo, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { apiFetch, readApiError, Tenant, VenueRegistrationCourt } from '@/lib/operator-api';
import { useRouter } from 'next/navigation';

const blankCourt = (sortOrder: number): VenueRegistrationCourt => ({
  code: `court-${String(sortOrder + 1).padStart(2, '0')}`,
  name: `${sortOrder + 1} 号场`,
  sort_order: sortOrder,
  status: 'active',
});

export default function AddVenueModal({ tenants }: { tenants: Tenant[] }) {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [tenantMode, setTenantMode] = useState<'existing' | 'new'>(tenants.length ? 'existing' : 'new');
  const [tenantId, setTenantId] = useState(tenants[0]?.id ?? '');
  const [tenantName, setTenantName] = useState('');
  const [venueCode, setVenueCode] = useState('');
  const [venueName, setVenueName] = useState('');
  const [address, setAddress] = useState('');
  const [courts, setCourts] = useState<VenueRegistrationCourt[]>([blankCourt(0)]);

  const canRemoveCourt = courts.length > 1;
  const tenantLabel = useMemo(
    () => tenants.find((tenant) => tenant.id === tenantId)?.name ?? '未选择租户',
    [tenantId, tenants],
  );

  function close() {
    if (loading) return;
    setIsOpen(false);
    setError('');
  }

  function updateCourt(index: number, field: keyof VenueRegistrationCourt, value: string) {
    setCourts((current) => current.map((court, itemIndex) => (
      itemIndex === index
        ? { ...court, [field]: field === 'sort_order' ? Number(value) : value }
        : court
    )));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    if (tenantMode === 'existing' && !tenantId) {
      setError('请选择一个已有租户，或切换为新建租户。');
      return;
    }
    setLoading(true);
    try {
      const response = await apiFetch('/api/v1/venue-registrations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantMode === 'existing' ? tenantId : null,
          tenant_name: tenantMode === 'new' ? tenantName : null,
          venue_code: venueCode,
          venue_name: venueName,
          timezone: 'Asia/Shanghai',
          address: address || null,
          courts,
        }),
      });
      if (!response.ok) throw new Error(await readApiError(response));
      setIsOpen(false);
      setVenueCode('');
      setVenueName('');
      setAddress('');
      setTenantName('');
      setCourts([blankCourt(0)]);
      router.refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '场馆注册失败。');
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <button onClick={() => setIsOpen(true)} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg font-medium transition-colors">
        注册球馆
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-slate-950/50 z-50 flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-labelledby="venue-registration-title">
          <div className="bg-white rounded-xl shadow-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto p-6">
            <div className="mb-6">
              <h2 id="venue-registration-title" className="text-xl font-bold text-slate-900">注册球馆与场地</h2>
              <p className="text-sm text-slate-500 mt-1">一次提交会创建完整归属关系：租户 → 球馆 → 至少一个场地。</p>
            </div>
            <form onSubmit={handleSubmit} className="space-y-6">
              <fieldset className="space-y-3">
                <legend className="text-sm font-semibold text-slate-800">所属租户</legend>
                <div className="flex gap-4 text-sm text-slate-700">
                  <label className="flex items-center gap-2"><input type="radio" checked={tenantMode === 'existing'} onChange={() => setTenantMode('existing')} /> 使用已有租户</label>
                  <label className="flex items-center gap-2"><input type="radio" checked={tenantMode === 'new'} onChange={() => setTenantMode('new')} /> 新建租户</label>
                </div>
                {tenantMode === 'existing' ? (
                  <select value={tenantId} onChange={(event) => setTenantId(event.target.value)} className="w-full border border-slate-300 rounded-lg px-3 py-2" required>
                    <option value="">选择租户</option>
                    {tenants.filter((tenant) => tenant.status === 'active').map((tenant) => <option key={tenant.id} value={tenant.id}>{tenant.name}</option>)}
                  </select>
                ) : (
                  <input required value={tenantName} onChange={(event) => setTenantName(event.target.value)} className="w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：好雨时节体育" />
                )}
                {tenantMode === 'existing' && <p className="text-xs text-slate-500">将注册到：{tenantLabel}</p>}
              </fieldset>

              <fieldset className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <legend className="text-sm font-semibold text-slate-800 mb-3">球馆信息</legend>
                <label className="text-sm text-slate-700">球馆名称<input required value={venueName} onChange={(event) => setVenueName(event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：好雨时节球馆" /></label>
                <label className="text-sm text-slate-700">球馆编码<input required value={venueCode} onChange={(event) => setVenueCode(event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：haoyushijie-01" /></label>
                <label className="md:col-span-2 text-sm text-slate-700">地址（可选）<input value={address} onChange={(event) => setAddress(event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" placeholder="例如：上海市徐汇区…" /></label>
              </fieldset>

              <fieldset>
                <div className="flex items-center justify-between mb-3"><legend className="text-sm font-semibold text-slate-800">首批场地</legend><button type="button" onClick={() => setCourts((current) => [...current, blankCourt(current.length)])} className="text-sm text-indigo-600 hover:text-indigo-800 font-medium inline-flex items-center gap-1"><Plus className="w-4 h-4" />增加场地</button></div>
                <div className="space-y-3">
                  {courts.map((court, index) => (
                    <div key={`${court.code}-${index}`} className="grid grid-cols-1 md:grid-cols-[1fr_1fr_120px_40px] gap-3 items-end rounded-lg border border-slate-200 p-3">
                      <label className="text-sm text-slate-700">场地名称<input required value={court.name} onChange={(event) => updateCourt(index, 'name', event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" /></label>
                      <label className="text-sm text-slate-700">场地编码<input required value={court.code} onChange={(event) => updateCourt(index, 'code', event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2" /></label>
                      <label className="text-sm text-slate-700">状态<select value={court.status} onChange={(event) => updateCourt(index, 'status', event.target.value)} className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2"><option value="active">可用</option><option value="maintenance">维护</option><option value="inactive">停用</option></select></label>
                      <button type="button" disabled={!canRemoveCourt} onClick={() => setCourts((current) => current.filter((_, itemIndex) => itemIndex !== index))} className="mb-1 p-2 text-slate-500 hover:text-rose-600 disabled:opacity-30" aria-label="删除场地"><Trash2 className="w-5 h-5" /></button>
                    </div>
                  ))}
                </div>
              </fieldset>

              {error && <p role="alert" className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
              <div className="flex justify-end gap-3 pt-2"><button type="button" onClick={close} className="px-4 py-2 text-slate-600 hover:bg-slate-100 rounded-lg">取消</button><button type="submit" disabled={loading} className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50">{loading ? '正在注册…' : '确认注册'}</button></div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
