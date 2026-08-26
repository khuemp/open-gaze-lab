/**
 * Main-thread side of the Pyodide worker: promise-based RPC plus progress.
 *
 * One worker is shared for the whole session, so the ~25 MB runtime is
 * downloaded once and both processing modes reuse it.
 */

let worker = null;
let nextRequestId = 1;
const pending = new Map();
const progressListeners = new Set();

function getWorker() {
    if (worker) return worker;

    // `new URL(..., import.meta.url)` is how Vite discovers, bundles and
    // base-path-rewrites a worker. The literal must stay inline — a variable
    // here would silently ship an unbundled path that 404s in production.
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

    worker.addEventListener('message', ({ data }) => {
        if (data.type === 'progress') {
            for (const listener of progressListeners) {
                listener({ stage: data.stage, detail: data.detail });
            }
            return;
        }

        const request = pending.get(data.id);
        if (!request) return;
        pending.delete(data.id);

        if (data.ok) request.resolve(data.result);
        else request.reject(new Error(data.error));
    });

    worker.addEventListener('error', (event) => {
        const message = event.message || 'The processing worker failed to start.';
        for (const request of pending.values()) request.reject(new Error(message));
        pending.clear();
    });

    return worker;
}

/**
 * Send one request to the worker.
 *
 * @param {string} cmd Command name understood by the worker.
 * @param {object} payload Structured-cloneable arguments.
 * @param {Transferable[]} transfer Buffers to move rather than copy — uploads
 *   can be hundreds of megabytes, and copying them would double peak memory.
 */
function call(cmd, payload = {}, transfer = []) {
    return new Promise((resolve, reject) => {
        const id = nextRequestId++;
        pending.set(id, { resolve, reject });
        getWorker().postMessage({ id, cmd, payload }, transfer);
    });
}

/** Subscribe to progress updates. Returns an unsubscribe function. */
export function onProgress(listener) {
    progressListeners.add(listener);
    return () => progressListeners.delete(listener);
}

/**
 * Start downloading the Python runtime.
 *
 * Called on mount so the download overlaps with the user choosing files and
 * filling in parameters, instead of starting when they press Process.
 */
export function warmUp() {
    return call('init');
}

export function runStationary(params) {
    const transfer = [params.csvBytes.buffer];
    if (params.backgroundImageBytes) transfer.push(params.backgroundImageBytes.buffer);
    return call('processStationary', params, transfer);
}

export function runHeadMounted(params) {
    return call('processHeadMounted', params, [params.zipBytes.buffer]);
}

/** Read a File into a Uint8Array for transfer to the worker. */
export async function readFileBytes(file) {
    return new Uint8Array(await file.arrayBuffer());
}
