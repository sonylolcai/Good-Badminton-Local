export const operatorApiBaseUrl =
  process.env.NEXT_PUBLIC_OPERATOR_API_BASE_URL ?? 'http://localhost:8000';

// Server-rendered pages use Docker's private service network rather than the
// public, Basic-Auth-protected reverse-proxy URL. Browser code must keep using
// `operatorApiBaseUrl` so it goes through the public operator API route.
export const operatorApiServerBaseUrl =
  process.env.OPERATOR_API_SERVER_BASE_URL ?? operatorApiBaseUrl;

export function apiFetch(path: string, init: RequestInit = {}) {
  return fetch(`${operatorApiBaseUrl}${path}`, { ...init, credentials: 'include' });
}

export interface AdminPrincipal {
  id: string;
  username: string;
  must_change_password: boolean;
  roles: { role: 'platform_admin' | 'venue_admin'; venue_id: string | null }[];
}

export function isPlatformAdmin(admin: AdminPrincipal) {
  return admin.roles.some((assignment) => assignment.role === 'platform_admin');
}

export type CourtStatus = 'active' | 'maintenance' | 'inactive';
export type CaptureMode = 'idle' | 'preview' | 'record';

export interface Tenant {
  id: string;
  name: string;
  status: 'active' | 'suspended';
}

export interface Court {
  id: string;
  venue_id: string;
  code: string;
  name: string;
  sort_order: number;
  status: CourtStatus;
}

export interface Venue {
  id: string;
  tenant_id: string;
  code: string;
  name: string;
  timezone: string;
  address: string | null;
  status: 'active' | 'inactive';
  court_count: number;
}

export interface VenueRegistrationCourt {
  code: string;
  name: string;
  sort_order: number;
  status: CourtStatus;
}

export interface ApiErrorPayload {
  error?: { code?: string; message?: string };
  detail?: { code?: string; message?: string } | string;
}

export interface CameraRuntime {
  connected: boolean;
  device_id: string | null;
  device_code: string | null;
  device_status: string;
  device_heartbeat_at: string | null;
  camera_id: string | null;
  camera_code: string | null;
  camera_status: string;
  camera_heartbeat_at: string | null;
  calibration_status: string;
}

export interface CaseRuntime {
  id: string;
  status: string;
  gpu_analysis_session_id: string | null;
  gpu_status: string | null;
  gpu_forwarding_enabled: boolean;
  preview_available: boolean;
  preview_url: string | null;
  last_preview_at: string | null;
  received_segment_count: number;
  forwarded_segment_count: number;
  last_received_at: string | null;
  last_forwarded_at: string | null;
  error: { code: string; message: string } | null;
}

export interface CaptureRuntime {
  mode: CaptureMode;
  revision: number;
  updated_at: string | null;
}

export interface CourtOperation {
  court: Pick<Court, 'id' | 'code' | 'name' | 'status'>;
  camera: CameraRuntime;
  capture: CaptureRuntime;
  case: CaseRuntime | null;
}

export interface GpuExecutionEvent {
  event_id?: string;
  event_type?: string;
  type?: string;
  level?: string;
  message?: string;
  source_time_sec?: number;
  occurred_at?: string;
  payload?: Record<string, unknown>;
}

export interface ReplayClip {
  id: string;
  case_id: string;
  start_segment_index: number;
  end_segment_index: number;
  segment_count: number;
  estimated_duration_seconds: number;
  url: string;
}

export interface VenueOperationsResponse {
  summary: { camera_connected: number; active_cases: number; total_courts: number };
  courts: CourtOperation[];
}

export async function readApiError(response: Response): Promise<string> {
  const payload = (await response.json().catch(() => ({}))) as ApiErrorPayload;
  if (typeof payload.detail === 'string') return payload.detail;
  return payload.error?.message ?? payload.detail?.message ?? '请求未完成，请稍后重试。';
}
