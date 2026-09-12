/**
 * Runs the backend test suite in the project's virtualenv, so `npm test` works
 * without the caller having to activate anything.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const apiDir = path.join(root, 'api');

const venvPython = process.platform === 'win32'
  ? path.join(apiDir, '.venv', 'Scripts', 'python.exe')
  : path.join(apiDir, '.venv', 'bin', 'python');

if (!fs.existsSync(venvPython)) {
  console.error('No virtualenv at api/.venv. Run "npm run setup" first.');
  process.exit(1);
}

const result = spawnSync(venvPython, ['-m', 'pytest', ...process.argv.slice(2)], {
  cwd: apiDir,
  stdio: 'inherit',
  shell: true,
});
process.exit(result.status ?? 1);
