import { cookies } from 'next/headers';
import { operatorApiServerBaseUrl } from './operator-api';

export async function operatorServerFetch(path: string, init: RequestInit = {}) {
  const cookie = (await cookies()).toString();
  return fetch(`${operatorApiServerBaseUrl}${path}`, {
    ...init,
    cache: 'no-store',
    headers: { ...init.headers, cookie },
  });
}
