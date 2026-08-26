/**
 * Vendors the pure-Python wheels Pyodide doesn't ship into public/wheels/.
 *
 * Only Plotly: its other runtime deps (narwhals, packaging) are already in
 * Pyodide's own package set and load via loadPackage(). Serving the wheel from
 * our own origin rather than letting micropip reach PyPI at run time keeps the
 * first load on one CDN and removes a third-party dependency from the critical
 * path.
 *
 * Skips the download when the file is already present, so repeated `npm run
 * dev` is cheap. Delete public/wheels/ to force a refresh.
 */
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WHEEL_DIR = join(ROOT, 'public', 'wheels');

// Keep in sync with PLOTLY_WHEEL in src/pyodide/worker.js.
const REQUIREMENTS = [{ name: 'plotly', version: '6.3.1' }];

async function resolveWheelUrl({ name, version }) {
    const response = await fetch(`https://pypi.org/pypi/${name}/${version}/json`);
    if (!response.ok) {
        throw new Error(`PyPI lookup failed for ${name}==${version}: ${response.status}`);
    }
    const { urls } = await response.json();
    const wheel = urls.find((u) => u.filename.endsWith('-py3-none-any.whl'));
    if (!wheel) {
        throw new Error(
            `${name}==${version} has no pure-Python wheel; Pyodide cannot install it.`,
        );
    }
    return wheel;
}

mkdirSync(WHEEL_DIR, { recursive: true });

for (const requirement of REQUIREMENTS) {
    const wheel = await resolveWheelUrl(requirement);
    const target = join(WHEEL_DIR, wheel.filename);

    if (existsSync(target)) {
        console.log(`[fetch-wheels] ${wheel.filename} already present, skipping`);
        continue;
    }

    const download = await fetch(wheel.url);
    if (!download.ok) {
        throw new Error(`Download failed for ${wheel.filename}: ${download.status}`);
    }
    writeFileSync(target, new Uint8Array(await download.arrayBuffer()));
    console.log(
        `[fetch-wheels] ${wheel.filename} (${(wheel.size / 1e6).toFixed(1)} MB)`,
    );
}
