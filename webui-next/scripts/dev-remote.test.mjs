import assert from 'node:assert/strict';
import test from 'node:test';

import { readRemoteEnv } from './dev-remote.mjs';

test('remote development config requires and normalizes the public operator API URL', () => {
  const values = readRemoteEnv('\n# remote operator\nNEXT_PUBLIC_OPERATOR_API_BASE_URL=https://operator.example.com/\n');
  assert.equal(values.NEXT_PUBLIC_OPERATOR_API_BASE_URL, 'https://operator.example.com');
  assert.throws(() => readRemoteEnv(''), /NEXT_PUBLIC_OPERATOR_API_BASE_URL/);
});
