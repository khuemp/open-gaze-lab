import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * Content hash of public/opengazelab.zip, written by scripts/pack-python.mjs.
 *
 * Assets in public/ are copied verbatim without Vite's filename hashing, so
 * the worker appends this as a query string to force a refetch when the Python
 * sources change. Missing during a bare `vite` run before pack-python.
 */
function pythonBundleHash() {
    try {
        return readFileSync(resolve(import.meta.dirname, '.python-bundle-hash'), 'utf-8').trim();
    } catch {
        return 'dev';
    }
}

export default defineConfig({
    // GitHub Pages serves project sites from /<repo>/, so every asset URL must
    // be prefixed. Overridable for other hosts (or '/' for a user/apex site).
    base: process.env.VITE_BASE ?? '/open-gaze-lab/',
    plugins: [react()],
    define: {
        __PY_BUNDLE_HASH__: JSON.stringify(pythonBundleHash()),
    },
    worker: {
        format: 'es',
    },
    build: {
        target: 'es2022',
        // Plotly HTML strings reach ~7 MB, so the informative warning here is
        // about our own chunks, not those runtime payloads.
        chunkSizeWarningLimit: 1500,
    },
});
