/**
 * One-command setup: Python venv + backend deps, then frontend deps.
 *
 * Finds a usable Python (py -3 on Windows, python3 elsewhere), creates
 * api/.venv if it is missing, and installs into it. Re-running is safe.
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const apiDir = path.join(root, 'api');
const webDir = path.join(root, 'web');

const venvPython = process.platform === 'win32'
  ? path.join(apiDir, '.venv', 'Scripts', 'python.exe')
  : path.join(apiDir, '.venv', 'bin', 'python');

const run = (command, args, cwd) => {
  console.log(`\n> ${command} ${args.join(' ')}`);
  const result = spawnSync(command, args, { cwd, stdio: 'inherit', shell: true });
  if (result.status !== 0) {
    console.error(`\nFailed: ${command} ${args.join(' ')}`);
    process.exit(result.status ?? 1);
  }
};

const findPython = () => {
  const candidates = process.platform === 'win32' ? ['py -3', 'python', 'python3'] : ['python3', 'python'];
  for (const candidate of candidates) {
    const [command, ...args] = candidate.split(' ');
    const probe = spawnSync(command, [...args, '--version'], { shell: true });
    if (probe.status === 0) return candidate;
  }
  console.error('No Python 3 found on PATH. Install Python 3.11+ and re-run.');
  process.exit(1);
};

if (!fs.existsSync(venvPython)) {
  const python = findPython();
  const [command, ...args] = python.split(' ');
  run(command, [...args, '-m', 'venv', '.venv'], apiDir);
} else {
  console.log('api/.venv already exists - reusing it.');
}

run(venvPython, ['-m', 'pip', 'install', '--upgrade', 'pip'], apiDir);
run(venvPython, ['-m', 'pip', 'install', '-r', 'requirements-dev.txt'], apiDir);
run('npm', ['install', '--no-audit', '--no-fund'], webDir);

console.log('\nSetup complete. Run "npm run dev" to start the API and the web app.');
