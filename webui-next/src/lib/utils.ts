import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * The business APIs exchange absolute UTC timestamps.  The operator console
 * must not depend on the browser/host timezone though: venue staff always see
 * China Standard Time, including when they use a machine outside China.
 */
export const CHINA_TIME_ZONE = 'Asia/Shanghai';

export function formatChinaTime(value: string | Date | null | undefined): string {
  if (!value) return '—';

  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);

  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: CHINA_TIME_ZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(date);
  const field = (type: Intl.DateTimeFormatPartTypes) => parts.find((part) => part.type === type)?.value ?? '';

  return `${field('year')}-${field('month')}-${field('day')} ${field('hour')}:${field('minute')}:${field('second')}`;
}
