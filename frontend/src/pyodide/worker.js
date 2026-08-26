/**
 * Pyodide worker — runs the whole OpenGazeLab pipeline off the main thread.
 *
 * Detection plus Plotly HTML generation takes long enough that doing it on the
 * main thread would freeze the tab, including the progress indicator that is
 * supposed to reassure the user during the ~25 MB first load. So all Python
 * lives here and talks to the page over a small request/response protocol:
 *
 *   in   { id, cmd, payload }
 *   out  { id, ok: true, result } | { id, ok: false, error }
 *   out  { type: 'progress', stage, detail }   (unsolicited, no id)
 *
 * Pyodide is pinned to 0.28.3 deliberately: it ships pandas 2.3.1 / numpy
 * 2.2.5 / scipy 1.14.1, matching what the pipeline was written and validated
 * against. Newer Pyodide releases ship pandas 3.x, whose copy-on-write
 * semantics change assignment behaviour throughout the pipeline. Do not bump
 * this without re-running the golden-file check in the README.
 */

const PYODIDE_VERSION = '0.28.3';
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Keep in sync with REQUIREMENTS in scripts/fetch-wheels.mjs.
const PLOTLY_WHEEL = 'wheels/plotly-6.3.1-py3-none-any.whl';

// Injected by vite.config.js from scripts/pack-python.mjs; busts the browser
// cache when the Python sources change (public/ assets aren't hashed).
const BUNDLE_HASH = typeof __PY_BUNDLE_HASH__ === 'string' ? __PY_BUNDLE_HASH__ : 'dev';

/**
 * Resolve an asset served from the site root (Vite's `public/` directory).
 *
 * Deliberately anchored to BASE_URL rather than `self.location`: the bundled
 * worker is emitted into assets/, so resolving relative to itself would look
 * for assets/opengazelab.zip. And a bare leading slash would 404 on GitHub
 * Pages, where the site lives under /open-gaze-lab/ rather than the domain
 * root. BASE_URL is the only thing that is correct in both dev and production.
 */
const asset = (path) =>
    new URL(path, new URL(import.meta.env.BASE_URL, self.location.href)).href;

let pyodide = null;
let webApi = null;
let bootPromise = null;
let scipyLoaded = false;

function reportProgress(stage, detail = '') {
    self.postMessage({ type: 'progress', stage, detail });
}

async function boot() {
    reportProgress('runtime', 'Downloading the Python runtime');
    const { loadPyodide } = await import(
        /* @vite-ignore */ `${PYODIDE_CDN}pyodide.mjs`
    );
    pyodide = await loadPyodide({ indexURL: PYODIDE_CDN });

    // narwhals + packaging are Plotly 6's runtime deps and ship with Pyodide,
    // so micropip never has to reach PyPI. SciPy is deliberately absent — it
    // is 13 MB and only the head-mounted path needs it (see ensureScipy).
    reportProgress('packages', 'Loading numpy, pandas and plotly');
    await pyodide.loadPackage(['numpy', 'pandas', 'micropip', 'narwhals', 'packaging']);

    const micropip = pyodide.pyimport('micropip');
    await micropip.install(asset(PLOTLY_WHEEL), { deps: false });
    micropip.destroy();

    reportProgress('package', 'Unpacking OpenGazeLab');
    const response = await fetch(asset(`opengazelab.zip?v=${BUNDLE_HASH}`));
    if (!response.ok) {
        throw new Error(`Could not load the OpenGazeLab Python bundle (${response.status})`);
    }
    pyodide.unpackArchive(await response.arrayBuffer(), 'zip');

    // Import once here rather than per request, so the first run isn't slowed
    // by module import and any packaging mistake surfaces during startup.
    webApi = pyodide.pyimport('opengazelab.web_api');

    reportProgress('ready', 'Ready');
}

function ensureBooted() {
    if (!bootPromise) {
        bootPromise = boot().catch((error) => {
            bootPromise = null; // let a later request retry a failed boot
            throw error;
        });
    }
    return bootPromise;
}

/** SciPy is only needed for Savitzky-Golay smoothing and GiW .mat loading. */
async function ensureScipy() {
    if (scipyLoaded) return;
    reportProgress('packages', 'Loading scipy for head-mounted processing');
    await pyodide.loadPackage('scipy');
    scipyLoaded = true;
}

/**
 * Call a function in opengazelab.web_api with keyword arguments.
 *
 * Arguments are converted and passed through a namespace rather than
 * interpolated into the source, so uploaded bytes stay bytes and nothing
 * user-supplied is ever parsed as Python.
 *
 * Null and undefined entries are dropped rather than passed through:
 * pyodide.toPy(null) yields a `jsnull` sentinel, not `None`, so an explicit
 * null would slip past every `is not None` guard in the pipeline. Letting the
 * Python signature's own default apply is both correct and simpler.
 */
async function callWebApi(functionName, kwargs) {
    const supplied = Object.fromEntries(
        Object.entries(kwargs).filter(([, value]) => value !== null && value !== undefined),
    );

    const namespace = pyodide.toPy({ kwargs: supplied });
    namespace.set('web_api', webApi);
    try {
        const result = await pyodide.runPythonAsync(
            `web_api.${functionName}(**kwargs)`,
            { globals: namespace, locals: namespace },
        );
        try {
            return result.toJs({ dict_converter: Object.fromEntries });
        } finally {
            result.destroy();
        }
    } finally {
        namespace.destroy();
    }
}

const COMMANDS = {
    /** Warm the runtime up front so the download overlaps with form filling. */
    async init() {
        await ensureBooted();
        return { ready: true };
    },

    async processStationary(payload) {
        await ensureBooted();
        reportProgress('detecting', 'Detecting fixations and saccades');
        return callWebApi('process_stationary', {
            csv_bytes: payload.csvBytes,
            resolution: payload.resolution,
            algorithm: payload.algorithm,
            sampling_rate: payload.samplingRate,
            min_fixation_duration: payload.minFixationDuration,
            detection_threshold: payload.detectionThreshold,
            y_origin: payload.yOrigin,
            fixation_merge_threshold: payload.fixationMergeThreshold ?? null,
            adapt: payload.adapt,
            bg_image_bytes: payload.backgroundImageBytes ?? null,
            bg_image_ext: payload.backgroundImageExt ?? null,
        });
    },

    async processHeadMounted(payload) {
        await ensureBooted();
        await ensureScipy();
        reportProgress('detecting', 'Detecting fixations and saccades');
        return callWebApi('process_head_mounted', {
            zip_bytes: payload.zipBytes,
            video_meta: payload.videoMeta,
            video_url: payload.videoUrl,
            resolution: payload.resolution,
            algorithm: payload.algorithm,
            sampling_rate: payload.samplingRate,
            min_fixation_duration: payload.minFixationDuration,
            detection_threshold: payload.detectionThreshold,
            adapt: payload.adapt,
            gain: payload.gain,
            window_size_ms: payload.windowSizeMs,
        });
    },
};

self.addEventListener('message', async ({ data }) => {
    const { id, cmd, payload } = data;
    try {
        const handler = COMMANDS[cmd];
        if (!handler) throw new Error(`Unknown worker command: ${cmd}`);

        const result = await handler(payload ?? {});
        reportProgress('idle', '');
        self.postMessage({ id, ok: true, result });
    } catch (error) {
        reportProgress('idle', '');
        self.postMessage({ id, ok: false, error: formatError(error) });
    }
});

/**
 * Python exceptions arrive with the full traceback in `message`. Surface only
 * the final "ValueError: ..." line, which is the part written for the user.
 */
function formatError(error) {
    const raw = error?.message ?? String(error);
    const match = raw.trim().match(/(?:^|\n)(?:\w+Error|Exception):\s*([\s\S]*)$/);
    return (match ? match[1] : raw).trim();
}
