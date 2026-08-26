/**
 * Bundles the Python package into a zip the Pyodide worker unpacks at startup.
 *
 * Everything under backend/opengazelab/ is included, not just *.py —
 * visualization/_video_template.html is read at import time by
 * video_overlay.py, so leaving it out breaks the head-mounted path.
 *
 * Also writes the bundle's content hash to .python-bundle-hash, which
 * vite.config.js injects as __PY_BUNDLE_HASH__ for cache-busting. The zip
 * filename itself is stable (it lives in public/, which Vite copies verbatim
 * without hashing), so the query string is what forces a refetch.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { zipSync } from 'fflate';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PACKAGE_DIR = resolve(ROOT, '..', 'backend', 'opengazelab');
const OUT_ZIP = join(ROOT, 'public', 'opengazelab.zip');
const OUT_HASH = join(ROOT, '.python-bundle-hash');

const EXCLUDED_DIRS = new Set(['__pycache__', '.pytest_cache', '.ruff_cache']);

/** Recursively collect files as { 'opengazelab/rel/path': Uint8Array }. */
function collect(dir, files = {}) {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            if (!EXCLUDED_DIRS.has(entry)) collect(full, files);
            continue;
        }
        if (entry.endsWith('.pyc')) continue;
        // Zip entries always use forward slashes, regardless of host OS.
        const key = `opengazelab/${relative(PACKAGE_DIR, full).split(/[\\/]/).join('/')}`;
        files[key] = new Uint8Array(readFileSync(full));
    }
    return files;
}

const files = collect(PACKAGE_DIR);
const names = Object.keys(files).sort();
if (!names.includes('opengazelab/web_api.py')) {
    throw new Error('web_api.py missing — the worker has no entry point to call.');
}
if (!names.includes('opengazelab/visualization/_video_template.html')) {
    throw new Error('_video_template.html missing — the video overlay will fail at import.');
}

// Sort entries and pin the timestamp so the same sources always produce the
// same bytes, keeping the hash (and the browser cache) stable across rebuilds.
// The ZIP format only encodes dates from 1980 onward, hence not the epoch.
const ordered = Object.fromEntries(names.map((n) => [n, files[n]]));
const zipped = zipSync(ordered, { level: 9, mtime: new Date('1980-01-01T00:00:00Z') });

mkdirSync(dirname(OUT_ZIP), { recursive: true });
writeFileSync(OUT_ZIP, zipped);

const hash = createHash('sha256').update(zipped).digest('hex').slice(0, 12);
writeFileSync(OUT_HASH, hash);

console.log(
    `[pack-python] ${names.length} files -> public/opengazelab.zip ` +
    `(${(zipped.length / 1024).toFixed(1)} KB, hash ${hash})`,
);
