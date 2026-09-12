/**
 * Runs the FastAPI backend and the web dev server together with prefixed,
 * interleaved output. A dev-only convenience, written by hand rather than
 * pulling in concurrently/npm-run-all for a few lines of spawn.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const venvPython = path.join(root, 'api', '.venv', 'Scripts', 'python.exe');
const venvPythonPosix = path.join(root, 'api', '.venv', 'bin', 'python');

const pythonExecutable = fs.existsSync(venvPython)
  ? venvPython
  : fs.existsSync(venvPythonPosix)
    ? venvPythonPosix
    : null;

if (!pythonExecutable) {
  console.error(
    'No virtualenv found at api/.venv. Run "npm run setup" first (it creates the venv and installs deps).',
  );
  process.exit(1);
}

const TARGETS = [
  {
    name: 'api',
    cwd: path.join(root, 'api'),
    command: pythonExecutable,
    args: ['-m', 'uvicorn', 'app.main:app', '--reload', '--port', '8787'],
    color: '[36m',
  },
  {
    name: 'web',
    cwd: path.join(root, 'web'),
    command: 'npm',
    args: ['run', 'dev'],
    color: '[35m',
  },
];

const RESET = '[0m';
const children = [];

const prefix = (target, stream) => (data) => {
  for (const line of data.toString().split('\n')) {
    if (line.trim()) stream.write(`${target.color}[${target.name}]${RESET} ${line}\n`);
  }
};

for (const target of TARGETS) {
  // shell:true because npm is a .cmd shim on Windows.
  const child = spawn(target.command, target.args, { cwd: target.cwd, shell: true });
  child.stdout.on('data', prefix(target, process.stdout));
  child.stderr.on('data', prefix(target, process.stderr));
  child.on('exit', (code) => {
    process.stdout.write(`${target.color}[${target.name}]${RESET} exited with code ${code}\n`);
    shutdown(code ?? 0);
  });
  children.push(child);
}

let shuttingDown = false;
function shutdown(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const child of children) child.kill();
  process.exit(code);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));
