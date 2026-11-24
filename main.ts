// Token Launch Radar entrypoint (Node 22+, standard library only).
// Reads /input/snapshot.json, runs the peer-calibrated valuation model in lib/model.ts,
// writes /output/board.json and /output/data.js, copies dashboard/ and screenshots/,
// and writes /output/dossier.json via vendor/makefun-board.ts.

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildRadar, parseSnapshot, renderDataScript, validateBoard } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const VALID_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

function flagValue(name: string, fallback: string): string {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx >= 0 && idx + 1 < process.argv.length) return process.argv[idx + 1]!;
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : fallback;
}

function copyDirFiles(src: string, dst: string): string[] {
  if (!existsSync(src)) return [];
  mkdirSync(dst, { recursive: true });
  const names = readdirSync(src).sort();
  const copied: string[] = [];
  for (const name of names) {
    if (name.startsWith('.')) continue;
    const full = join(src, name);
    if (!statSync(full).isFile()) continue;
    if (!VALID_NAME.test(name)) {
      throw new Error(`invalid file name in ${src}: ${name}`);
    }
    copyFileSync(full, join(dst, name));
    copied.push(name);
  }
  return copied;
}

const inputArg = flagValue('input', '/input');
const inputPath = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputPath)) {
  process.stderr.write(`token-launch-radar: snapshot not found at ${inputPath}\n`);
  process.exit(2);
}

const outDir = flagValue('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');
const snapshot = parseSnapshot(readFileSync(inputPath, 'utf8'));
const { board, dataset } = buildRadar(snapshot);

const err = validateBoard(board, snapshot);
if (err !== null) {
  process.stderr.write(`token-launch-radar: gate check failed: ${err}\n`);
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
writeFileSync(join(outDir, 'data.js'), renderDataScript(dataset));

const dashFiles = copyDirFiles(join(ROOT, 'dashboard'), join(outDir, 'dashboard'));
const shotFiles = copyDirFiles(join(ROOT, 'screenshots'), join(outDir, 'screenshots'));

if (shotFiles.length < 3 || shotFiles.length > 4) {
  process.stderr.write(`token-launch-radar: expected 3-4 screenshots, found ${shotFiles.length}\n`);
  process.exit(1);
}

const dossier = buildDossier(outDir, {
  text_max_bytes: TEXT_MAX_BYTES,
  screenshots: SCREENSHOT_CAPS,
  dashboard: { max_files: 400, max_bytes: 4_194_304 },
});
writeFileSync(join(outDir, 'dossier.json'), dossier);

process.stdout.write(
  `token-launch-radar: ${dataset.universe.board_count} pre-token rows (${dataset.universe.watchlist_count} watchlist, ${dataset.universe.early_count} early) calibrated from ${dataset.calibration.live_peer_count} live comps (median ${dataset.calibration.global_median}x)\n` +
    `token-launch-radar: wrote board.json, data.js, dashboard (${dashFiles.length} files), screenshots (${shotFiles.length} images), dossier.json (${Buffer.byteLength(dossier)} bytes)\n`,
);
