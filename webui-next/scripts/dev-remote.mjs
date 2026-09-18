import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ENV_FILE = resolve('.env.remote.local');

export function readRemoteEnv(text) {
  const values = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const separator = line.indexOf('=');
    if (separator < 1) throw new Error(`Invalid remote environment entry: ${rawLine}`);
    const key = line.slice(0, separator).trim();
    let value = line.slice(separator + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[key] = value;
  }
  const apiUrl = values.NEXT_PUBLIC_OPERATOR_API_BASE_URL;
  if (!apiUrl) throw new Error('NEXT_PUBLIC_OPERATOR_API_BASE_URL is required in .env.remote.local');
  const parsed = new URL(apiUrl);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('NEXT_PUBLIC_OPERATOR_API_BASE_URL must use http or https');
  }
  values.NEXT_PUBLIC_OPERATOR_API_BASE_URL = parsed.toString().replace(/\/$/, '');
  return values;
}

function main() {
  if (!existsSync(ENV_FILE)) {
    throw new Error('Missing .env.remote.local. Copy .env.remote.example and set the remote HTTPS API origin.');
  }
  const remoteEnv = readRemoteEnv(readFileSync(ENV_FILE, 'utf8'));
  const require = createRequire(import.meta.url);
  const nextBin = require.resolve('next/dist/bin/next');
  process.stdout.write(`Using remote Operator API: ${remoteEnv.NEXT_PUBLIC_OPERATOR_API_BASE_URL}\n`);
  const child = spawn(process.execPath, [nextBin, 'dev', ...process.argv.slice(2)], {
    stdio: 'inherit',
    env: { ...process.env, ...remoteEnv },
  });
  child.on('exit', (code, signal) => process.exitCode = code ?? (signal ? 1 : 0));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
