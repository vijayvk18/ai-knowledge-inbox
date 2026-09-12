/**
 * Runs the FastAPI backend alone, resolving the venv on either platform so
 * `npm run dev:api` works without activating anything.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const apiDir = path.join(root, 'api');

const candidates = [
  path.join(apiDir, '.venv', 'Scripts', 'python.exe'),
  path.join(apiDir, '.venv', 'bin', 'python'),
];
const python = candidates.find((candidate) => fs.existsSync(candidate));

if (!python) {
  console.error('No virtualenv at api/.venv. Run "npm run setup" first.');
  process.exit(1);
}

const args = ['-m', 'uvicorn', 'app.main:app', '--port', '8787'];
if (!process.argv.includes('--no-reload')) args.push('--reload');

const result = spawnSync(python, args, { cwd: apiDir, stdio: 'inherit', shell: true });
process.exit(result.status ?? 1);
