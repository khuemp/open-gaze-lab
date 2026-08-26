import React, { useCallback, useEffect, useRef, useState } from 'react';

import {
    onProgress,
    readFileBytes,
    runHeadMounted,
    runStationary,
    warmUp,
} from './pyodide/client.js';
import { probeVideoMetadata } from './video/probeVideo.js';

// NumericInput component that only accepts numbers
function NumericInput({ id, value, onChange, placeholder, className, allowDecimal = true, allowNegative = false }) {
    const handleKeyDown = (e) => {
        // Allow: backspace, delete, tab, escape, enter, arrows
        const allowedKeys = ['Backspace', 'Delete', 'Tab', 'Escape', 'Enter', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
        if (allowedKeys.includes(e.key)) return;

        // Allow Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+X
        if ((e.ctrlKey || e.metaKey) && ['a', 'c', 'v', 'x'].includes(e.key.toLowerCase())) return;

        // Allow decimal point (only one)
        if (allowDecimal && e.key === '.' && !value.includes('.')) return;

        // Allow negative sign (only at start)
        if (allowNegative && e.key === '-' && !value.includes('-') && e.target.selectionStart === 0) return;

        // Allow digits
        if (/^\d$/.test(e.key)) return;

        // Block everything else
        e.preventDefault();
    };

    const handleChange = (e) => {
        const nextValue = e.target.value;
        let pattern = allowDecimal ? /^\d*\.?\d*$/ : /^\d*$/;
        if (allowNegative) pattern = allowDecimal ? /^-?\d*\.?\d*$/ : /^-?\d*$/;

        if (pattern.test(nextValue)) {
            onChange(nextValue);
        }
    };

    return (
        <input
            id={id}
            type="text"
            inputMode="decimal"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={className}
        />
    );
}

// ---------------------------------------------------------------------------
// Parameter parsing
//
// The FastAPI layer used to validate these server-side and return HTTP 400.
// With the pipeline running in-page there is no request to reject, so the same
// checks happen here and surface through the existing error banner.
// ---------------------------------------------------------------------------

/** Throw a single message naming every empty required field. */
function requireFields(fields) {
    const missing = Object.entries(fields)
        .filter(([, value]) => value === '' || value === null || value === undefined)
        .map(([label]) => label);
    if (missing.length > 0) {
        throw new Error(`Please fill in: ${missing.join(', ')}`);
    }
}

function parseResolution(text) {
    const parts = text.split(',').map((part) => Number(part.trim()));
    if (parts.length !== 2 || parts.some((n) => !Number.isFinite(n) || n <= 0)) {
        throw new Error("Resolution must be 'width,height' — for example 1920,1080");
    }
    return parts;
}

/** Optional numeric field: '' means "not set" and stays null. */
const optionalNumber = (value) => (value === '' ? null : Number(value));

/** Filename without its extension, used to name the downloaded CSV. */
const fileStem = (name) => name.replace(/\.[^./\\]+$/, '');

const fileExtension = (name) => {
    const match = name.match(/\.([^./\\]+)$/);
    return match ? match[1].toLowerCase() : 'png';
};

/**
 * Per-algorithm guidance for the Detection Threshold box.
 *
 * The same field means two different quantities — a dispersion in pixels for
 * I-DT, a velocity in px/ms for I-VT — and the values differ by more than an
 * order of magnitude. A fixed placeholder invites entering an I-DT number
 * while I-VT is selected, which classifies every sample as a fixation and
 * silently reports zero saccades rather than failing.
 *
 * The head-mounted numbers are the per-algorithm bests from the DD parameter
 * sweep documented in the README. No equivalent sweep exists for the
 * stationary I-VT path, so that one carries units only rather than an
 * invented figure.
 */
const THRESHOLD_GUIDANCE = {
    stationary: {
        idt: { placeholder: '125', hint: 'Dispersion, in pixels' },
        ivt: { placeholder: '', hint: 'Velocity, in px/ms' },
        '': { placeholder: '', hint: 'Units depend on the algorithm' },
    },
    headmounted: {
        idt: { placeholder: '30', hint: 'Relative dispersion, in pixels' },
        ivt: { placeholder: '1.5', hint: 'Relative velocity, in px/ms' },
        '': { placeholder: '', hint: 'Units depend on the algorithm' },
    },
};

const thresholdGuidance = (mode, algorithm) =>
    THRESHOLD_GUIDANCE[mode][algorithm] ?? THRESHOLD_GUIDANCE[mode][''];

const STAGE_LABELS = {
    runtime: 'Downloading the Python runtime (one-time, then cached)',
    packages: 'Loading scientific packages',
    package: 'Unpacking OpenGazeLab',
    detecting: 'Detecting fixations and saccades',
    ready: 'Ready',
    idle: '',
};

function App() {
    // Mode toggle
    const [mode, setMode] = useState('stationary'); // 'stationary' | 'headmounted'

    // Stationary mode state
    const [file, setFile] = useState(null);
    const [backgroundImage, setBackgroundImage] = useState(null);
    const [resolution, setResolution] = useState('');
    const [minFixationDuration, setMinFixationDuration] = useState('');
    const [detectionThreshold, setDetectionThreshold] = useState('');
    const [algorithm, setAlgorithm] = useState('');
    const [samplingRate, setSamplingRate] = useState('');
    const [fixationMergeThreshold, setFixationMergeThreshold] = useState('');
    const [adapt, setAdapt] = useState(false);
    const [yOrigin, setYOrigin] = useState('');

    // Head-mounted mode state
    const [datasetZip, setDatasetZip] = useState(null);
    const [videoFile, setVideoFile] = useState(null);
    const [hmResolution, setHmResolution] = useState('');
    const [hmMinFixation, setHmMinFixation] = useState('');
    const [hmThreshold, setHmThreshold] = useState('');
    const [hmAlgorithm, setHmAlgorithm] = useState('');
    const [hmSamplingRate, setHmSamplingRate] = useState('');
    const [hmAdapt, setHmAdapt] = useState(false);
    const [hmGain, setHmGain] = useState('');
    const [hmWindowSizeMs, setHmWindowSizeMs] = useState('');
    const [hmFps, setHmFps] = useState('');

    // Video probing — replaces what cv2 used to read off the file server-side
    const [videoMeta, setVideoMeta] = useState(null);
    const [probing, setProbing] = useState(false);

    // Results / status
    const [stationaryResults, setStationaryResults] = useState(null);
    const [hmResults, setHmResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [stage, setStage] = useState('runtime');

    // The overlay player streams the scene video straight from the user's disk
    // through a blob URL. It has to outlive the request that produced the
    // results HTML, so it is revoked only when replaced or on unmount.
    const videoUrlRef = useRef(null);

    const releaseVideoUrl = useCallback(() => {
        if (videoUrlRef.current) {
            URL.revokeObjectURL(videoUrlRef.current);
            videoUrlRef.current = null;
        }
    }, []);

    useEffect(() => onProgress(({ stage: next }) => setStage(next)), []);

    // Start the ~25 MB runtime download immediately, so it overlaps with the
    // user picking files and filling in parameters rather than beginning when
    // they press Process.
    useEffect(() => {
        warmUp().catch((err) => setError(`Could not start the Python runtime: ${err.message}`));
    }, []);

    useEffect(() => releaseVideoUrl, [releaseVideoUrl]);

    const handleBackgroundImageSelect = (selectedFile) => {
        setBackgroundImage(selectedFile);
        setError(null);
    };

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setError(null);
    };

    /** Probe fps/resolution as soon as a video is chosen, not at process time. */
    const handleVideoSelect = async (selectedFile) => {
        setVideoFile(selectedFile);
        setError(null);
        setVideoMeta(null);

        if (!selectedFile) return;

        setProbing(true);
        try {
            const meta = await probeVideoMetadata(selectedFile);
            setVideoMeta(meta);
            if (meta.fps > 0) setHmFps(String(Number(meta.fps.toFixed(3))));
            if (meta.width > 0 && meta.height > 0) {
                setHmResolution(`${meta.width},${meta.height}`);
            }
            if (meta.warning) setError(meta.warning);
        } finally {
            setProbing(false);
        }
    };

    const handleModeSwitch = (newMode) => {
        setMode(newMode);
        setError(null);
    };

    const handleProcessHeadMounted = async () => {
        requireFields({
            'Screen Resolution': hmResolution,
            'Sampling Rate': hmSamplingRate,
            'Min Fixation Duration': hmMinFixation,
            'Detection Threshold': hmThreshold,
            Algorithm: hmAlgorithm,
            'Video FPS': hmFps,
        });

        const [width, height] = parseResolution(hmResolution);
        const fps = Number(hmFps);
        if (!Number.isFinite(fps) || fps <= 0) {
            throw new Error('Video FPS must be greater than zero.');
        }

        const durationSeconds = videoMeta?.duration_s ?? 0;
        const zipBytes = await readFileBytes(datasetZip);

        releaseVideoUrl();
        videoUrlRef.current = URL.createObjectURL(videoFile);

        const result = await runHeadMounted({
            zipBytes,
            videoMeta: {
                fps,
                width,
                height,
                duration_s: durationSeconds,
                // Prefer the exact container frame count; derive one only when
                // the probe could not supply it or the user overrode the fps.
                n_frames: videoMeta?.n_frames || Math.round(fps * durationSeconds),
            },
            videoUrl: videoUrlRef.current,
            resolution: [width, height],
            algorithm: hmAlgorithm,
            samplingRate: Number(hmSamplingRate),
            minFixationDuration: Number(hmMinFixation),
            detectionThreshold: Number(hmThreshold),
            adapt: hmAdapt,
            gain: hmGain === '' ? 0 : Number(hmGain),
            windowSizeMs: hmWindowSizeMs === '' ? 0 : Number(hmWindowSizeMs),
        });

        setHmResults({ ...result, filename: fileStem(videoFile.name) });
    };

    const handleProcessStationary = async () => {
        requireFields({
            'Screen Resolution': resolution,
            'Sampling Rate': samplingRate,
            'Minimal Fixation Duration': minFixationDuration,
            'Detection Threshold': detectionThreshold,
            Algorithm: algorithm,
            'Y-Origin': yOrigin,
        });

        const [width, height] = parseResolution(resolution);

        const result = await runStationary({
            csvBytes: await readFileBytes(file),
            resolution: [width, height],
            algorithm,
            samplingRate: Number(samplingRate),
            minFixationDuration: Number(minFixationDuration),
            detectionThreshold: Number(detectionThreshold),
            yOrigin,
            fixationMergeThreshold: optionalNumber(fixationMergeThreshold),
            adapt,
            backgroundImageBytes: backgroundImage ? await readFileBytes(backgroundImage) : null,
            backgroundImageExt: backgroundImage ? fileExtension(backgroundImage.name) : null,
        });

        setStationaryResults({ ...result, filename: fileStem(file.name) });
    };

    const handleProcess = async () => {
        setLoading(true);
        setError(null);

        try {
            if (mode === 'headmounted') {
                setHmResults(null);
                await handleProcessHeadMounted();
            } else {
                setStationaryResults(null);
                await handleProcessStationary();
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error processing gaze data');
        } finally {
            setLoading(false);
        }
    };

    const stageLabel = STAGE_LABELS[stage] ?? '';

    return (
        <div className="container">
            <header className="header">
                <h1>OpenGazeLab</h1>
                <div className="mode-toggle">
                    <button
                        className={`mode-btn ${mode === 'stationary' ? 'active' : ''}`}
                        onClick={() => handleModeSwitch('stationary')}
                    >
                        Stationary Eye Tracker
                    </button>
                    <button
                        className={`mode-btn ${mode === 'headmounted' ? 'active' : ''}`}
                        onClick={() => handleModeSwitch('headmounted')}
                    >
                        Head-Mounted Eye Tracker
                    </button>
                </div>
                <p className="privacy-note">
                    Everything runs in your browser. Your recordings are never uploaded.
                </p>
            </header>

            <main className="main-content">
                {mode === 'stationary' ? (
                    <React.Fragment>
                        <div className="panel">
                            <div className="upload-row">
                                <Upload
                                    title="Gaze Data"
                                    accept=".csv"
                                    validate={(f) => f.type === 'text/csv' || f.name.endsWith('.csv')}
                                    errorMsg="Please select a CSV file"
                                    icon={ICON_UPLOAD}
                                    inputId="csv-input"
                                    fileName={file?.name}
                                    onFileSelect={handleFileSelect}
                                    placeholderText="Select CSV file"
                                    selectedText="File ready for processing"
                                />
                                <Upload
                                    title="Background Image"
                                    accept="image/*"
                                    validate={(f) => f.type.startsWith('image/')}
                                    errorMsg="Please select an image file (PNG, JPG, etc.)"
                                    icon={ICON_IMAGE}
                                    inputId="bg-image-input"
                                    fileName={backgroundImage?.name}
                                    onFileSelect={handleBackgroundImageSelect}
                                    placeholderText="Select background image"
                                    selectedText="Image will be shown behind plots"
                                    extraClass="background-image-upload"
                                    selectedExtra={{
                                        className: 'bg-image-selected',
                                        button: (
                                            <button
                                                onClick={(e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    handleBackgroundImageSelect(null);
                                                }}
                                                className="clear-image-btn"
                                            >
                                                Clear
                                            </button>
                                        ),
                                    }}
                                />
                            </div>
                        </div>

                        <div className="panel">
                            <h2>Detection Parameters</h2>
                            <div className="params-grid">
                                <div className="control-group">
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <div style={{ flex: 1 }}>
                                            <label htmlFor="algorithm">Algorithm</label>
                                            <select
                                                id="algorithm"
                                                value={algorithm}
                                                onChange={(e) => setAlgorithm(e.target.value)}
                                                className="input-field"
                                            >
                                                <option value="">Select algorithm</option>
                                                <option value="idt">I-DT</option>
                                                <option value="ivt">I-VT</option>
                                            </select>
                                        </div>
                                        <div style={{ flex: 1 }}>
                                            <label htmlFor="y-origin">Y-Origin</label>
                                            <select
                                                id="y-origin"
                                                value={yOrigin}
                                                onChange={(e) => setYOrigin(e.target.value)}
                                                className="input-field"
                                            >
                                                <option value="">Select origin</option>
                                                <option value="top-left">Top-Left</option>
                                                <option value="top-right">Top-Right</option>
                                                <option value="bottom-left">Bottom-Left</option>
                                                <option value="bottom-right">Bottom-Right</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="control-group">
                                    <label htmlFor="resolution">Screen Resolution (W,H)</label>
                                    <input
                                        id="resolution"
                                        type="text"
                                        value={resolution}
                                        onChange={(e) => setResolution(e.target.value)}
                                        placeholder="2560,1440"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="sampling-rate">Sampling Rate (Hz)</label>
                                    <NumericInput
                                        id="sampling-rate"
                                        value={samplingRate}
                                        onChange={setSamplingRate}
                                        placeholder="250"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="min-fixation">Minimal Fixation Duration (ms)</label>
                                    <NumericInput
                                        id="min-fixation"
                                        value={minFixationDuration}
                                        onChange={setMinFixationDuration}
                                        placeholder="50"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="detect-threshold">Detection Threshold</label>
                                    <NumericInput
                                        id="detect-threshold"
                                        value={detectionThreshold}
                                        onChange={setDetectionThreshold}
                                        placeholder={thresholdGuidance('stationary', algorithm).placeholder}
                                        className="input-field"
                                    />
                                    <p className="field-hint">
                                        {thresholdGuidance('stationary', algorithm).hint}
                                    </p>
                                </div>

                                <div className="control-group">
                                    <label htmlFor="fixation-merge">Merge Threshold (px)</label>
                                    <NumericInput
                                        id="fixation-merge"
                                        value={fixationMergeThreshold}
                                        onChange={setFixationMergeThreshold}
                                        placeholder="None"
                                        className="input-field"
                                    />
                                </div>
                            </div>

                            <div className="control-group" style={{ marginBottom: 0 }}>
                                <label htmlFor="adapt">
                                    <input
                                        id="adapt"
                                        type="checkbox"
                                        checked={adapt}
                                        onChange={(e) => setAdapt(e.target.checked)}
                                        style={{ marginRight: '8px' }}
                                    />
                                    Enable Adaptive Threshold
                                </label>
                            </div>

                            <button
                                className="process-button"
                                onClick={handleProcess}
                                disabled={loading || !file}
                            >
                                {loading ? 'Processing...' : 'Process Gaze Data'}
                            </button>
                        </div>
                    </React.Fragment>
                ) : (
                    <React.Fragment>
                        <div className="panel">
                            <div className="upload-row">
                                <Upload
                                    title="Dataset"
                                    accept=".zip"
                                    validate={(f) => f.name.toLowerCase().endsWith('.zip')}
                                    errorMsg="Please select a .zip file"
                                    icon={ICON_UPLOAD}
                                    inputId="zip-input"
                                    fileName={datasetZip?.name}
                                    onFileSelect={setDatasetZip}
                                    placeholderText="Select ZIP dataset"
                                    hintText="Drews & Dierkes or Gaze-in-Wild layout"
                                    selectedText="Dataset ready"
                                />
                                <Upload
                                    title="Scene Video"
                                    accept=".mp4,video/mp4"
                                    validate={(f) => f.name.toLowerCase().endsWith('.mp4')}
                                    errorMsg="Please select an .mp4 video file"
                                    icon={ICON_VIDEO}
                                    inputId="video-input"
                                    fileName={videoFile?.name}
                                    onFileSelect={handleVideoSelect}
                                    placeholderText="Select MP4 video"
                                    hintText="Scene camera recording"
                                    selectedText={
                                        probing
                                            ? 'Reading video metadata...'
                                            : videoMeta
                                                ? `${videoMeta.width}x${videoMeta.height}, `
                                                  + `${videoMeta.fps.toFixed(2)} fps, `
                                                  + `${videoMeta.n_frames} frames`
                                                : 'Video ready'
                                    }
                                />
                            </div>
                        </div>

                        <div className="panel">
                            <h2>Detection Parameters</h2>
                            <div className="params-grid">
                                <div className="control-group">
                                    <label htmlFor="hm-algorithm">Algorithm</label>
                                    <select
                                        id="hm-algorithm"
                                        value={hmAlgorithm}
                                        onChange={(e) => setHmAlgorithm(e.target.value)}
                                        className="input-field"
                                    >
                                        <option value="">Select algorithm</option>
                                        <option value="idt">I-DT</option>
                                        <option value="ivt">I-VT</option>
                                    </select>
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-resolution">Screen Resolution (W,H)</label>
                                    <input
                                        id="hm-resolution"
                                        type="text"
                                        value={hmResolution}
                                        onChange={(e) => setHmResolution(e.target.value)}
                                        placeholder="1088,1080"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-fps">Video FPS</label>
                                    <NumericInput
                                        id="hm-fps"
                                        value={hmFps}
                                        onChange={setHmFps}
                                        placeholder="25"
                                        className="input-field"
                                    />
                                    <p className="field-hint">
                                        {videoMeta?.source === 'mp4box'
                                            ? 'Read from the video container'
                                            : 'Enter the scene camera frame rate'}
                                    </p>
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-sampling-rate">Sampling Rate (Hz)</label>
                                    <NumericInput
                                        id="hm-sampling-rate"
                                        value={hmSamplingRate}
                                        onChange={setHmSamplingRate}
                                        placeholder="200"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-min-fixation">Min Fixation Duration (ms)</label>
                                    <NumericInput
                                        id="hm-min-fixation"
                                        value={hmMinFixation}
                                        onChange={setHmMinFixation}
                                        placeholder="54"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-threshold">Detection Threshold</label>
                                    <NumericInput
                                        id="hm-threshold"
                                        value={hmThreshold}
                                        onChange={setHmThreshold}
                                        placeholder={thresholdGuidance('headmounted', hmAlgorithm).placeholder}
                                        className="input-field"
                                    />
                                    <p className="field-hint">
                                        {thresholdGuidance('headmounted', hmAlgorithm).hint}
                                    </p>
                                </div>

                                <div className="control-group threshold-extras">
                                    <div className="threshold-extra-item">
                                        <label htmlFor="hm-gain">Gain</label>
                                        <NumericInput
                                            id="hm-gain"
                                            value={hmGain}
                                            onChange={setHmGain}
                                            placeholder="0"
                                            className="input-field"
                                        />
                                    </div>
                                    <div className="threshold-extra-item">
                                        <label htmlFor="hm-window-size">Window Size (ms)</label>
                                        <NumericInput
                                            id="hm-window-size"
                                            value={hmWindowSizeMs}
                                            onChange={setHmWindowSizeMs}
                                            placeholder="0"
                                            className="input-field"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="control-group" style={{ marginBottom: 0 }}>
                                <label htmlFor="hm-adapt">
                                    <input
                                        id="hm-adapt"
                                        type="checkbox"
                                        checked={hmAdapt}
                                        onChange={(e) => setHmAdapt(e.target.checked)}
                                        style={{ marginRight: '8px' }}
                                    />
                                    Enable Adaptive Threshold
                                </label>
                            </div>

                            <button
                                className="process-button"
                                onClick={handleProcess}
                                disabled={loading || probing || !datasetZip || !videoFile}
                            >
                                {loading ? 'Processing...' : 'Process Gaze Data'}
                            </button>
                        </div>
                    </React.Fragment>
                )}

                {error && <div className="error-message">{error}</div>}

                {loading && (
                    <div className="loading-spinner">
                        {stageLabel || 'Processing...'}
                    </div>
                )}

                {!loading && stage !== 'idle' && stage !== 'ready' && (
                    <div className="runtime-status">{stageLabel}</div>
                )}

                {stationaryResults && mode === 'stationary' && (
                    <ResultsDisplay results={stationaryResults} variant="stationary" />
                )}
                {hmResults && mode === 'headmounted' && (
                    <ResultsDisplay results={hmResults} variant="headmounted" />
                )}
            </main>
        </div>
    );
}

const ICON_UPLOAD = (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
);

const ICON_IMAGE = (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <polyline points="21 15 16 10 5 21" />
    </svg>
);

const ICON_VIDEO = (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
        <line x1="7" y1="2" x2="7" y2="22" />
        <line x1="17" y1="2" x2="17" y2="22" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <line x1="2" y1="7" x2="7" y2="7" />
        <line x1="2" y1="17" x2="7" y2="17" />
        <line x1="17" y1="7" x2="22" y2="7" />
        <line x1="17" y1="17" x2="22" y2="17" />
    </svg>
);

function Upload({
    title,
    accept,
    validate,
    errorMsg,
    icon,
    inputId,
    fileName,
    onFileSelect,
    placeholderText,
    hintText = 'or drag and drop',
    selectedText,
    extraClass = '',
    selectedExtra = null,
}) {
    const [isDragging, setIsDragging] = useState(false);

    const handleFile = (file) => {
        if (validate(file)) onFileSelect(file);
        else alert(errorMsg);
    };

    const handleFileChange = (e) => {
        const f = e.target.files?.[0];
        if (f) handleFile(f);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f) handleFile(f);
    };

    return (
        <div className={`file-upload ${extraClass}`.trim()}>
            <h2>{title}</h2>
            <div
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept={accept}
                    onChange={handleFileChange}
                    id={inputId}
                    className="hidden-input"
                />
                <label htmlFor={inputId} className="upload-label">
                    <div className="upload-icon">{icon}</div>
                    <p className="upload-text">
                        {fileName ? `Selected: ${fileName}` : placeholderText}
                    </p>
                    <p className="upload-hint">{hintText}</p>
                </label>
            </div>
            {fileName && (
                <div className={`file-selected ${selectedExtra?.className || ''}`.trim()}>
                    {selectedText}
                    {selectedExtra?.button}
                </div>
            )}
        </div>
    );
}

/**
 * Turn a string produced in Python into an object URL for this render.
 *
 * The plots reach ~7 MB, which is too much for an iframe `srcdoc` attribute,
 * so each becomes a blob served to the iframe instead. Blob documents inherit
 * this page's origin, so the video overlay can still load the scene video's
 * own blob URL and postMessage its height back to us.
 */
function useBlobUrl(content, type) {
    const [url, setUrl] = useState(null);

    useEffect(() => {
        if (!content) {
            setUrl(null);
            return undefined;
        }
        const objectUrl = URL.createObjectURL(new Blob([content], { type }));
        setUrl(objectUrl);
        return () => URL.revokeObjectURL(objectUrl);
    }, [content, type]);

    return url;
}

function ResultsDisplay({ results, variant = 'stationary' }) {
    const r = results.summary || {};
    const isHm = variant === 'headmounted';
    const [videoFrameHeight, setVideoFrameHeight] = useState(1000);

    const csvUrl = useBlobUrl(results.events_csv, 'text/csv');
    const plotUrl = useBlobUrl(results.plot_html, 'text/html');
    const timePlotUrl = useBlobUrl(results.time_plot_html, 'text/html');
    const videoPlotUrl = useBlobUrl(results.video_plot_html, 'text/html');

    useEffect(() => {
        if (!isHm) return undefined;
        const handleMessage = (event) => {
            if (event.data?.type !== 'video-gaze-visualization-height') return;
            const nextHeight = Number(event.data.height);
            if (Number.isFinite(nextHeight) && nextHeight > 0) {
                setVideoFrameHeight((cur) => {
                    const rounded = Math.ceil(nextHeight);
                    return Math.abs(cur - rounded) > 4 ? rounded : cur;
                });
            }
        };
        window.addEventListener('message', handleMessage);
        return () => window.removeEventListener('message', handleMessage);
    }, [isHm]);

    return (
        <div className="results-container">
            <div className="results-header">
                <h2>Detection Results</h2>
                {csvUrl && (
                    <a
                        href={csvUrl}
                        className="download-button"
                        download={`${results.filename}_events.csv`}
                    >
                        Download Events CSV
                    </a>
                )}
            </div>

            <div className={`results-grid ${isHm ? 'hm-results-grid' : ''}`.trim()}>
                <div className="result-card">
                    <p className="result-label">Total Gaze Samples</p>
                    <p className="result-value">{r.num_events || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Fixation Samples</p>
                    <p className="result-value">{r.num_fixations || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Saccade Samples</p>
                    <p className="result-value">{r.num_saccades || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Fixation Events</p>
                    <p className="result-value">{r.num_fixation_points || 0}</p>
                </div>
                {!isHm && (
                    <React.Fragment>
                        <div className="result-card">
                            <p className="result-label">Out of Range</p>
                            <p className="result-value">{r.num_oor_gaze_points || 0}</p>
                        </div>
                        <div className="result-card">
                            <p className="result-label">Invalid (NaN)</p>
                            <p className="result-value">{r.num_nan_gaze_points || 0}</p>
                        </div>
                    </React.Fragment>
                )}
                {isHm && r.f1_fixation != null && (
                    <div className="result-card">
                        <p className="result-label">F1 Fixation</p>
                        <p className="result-value">{r.f1_fixation}</p>
                    </div>
                )}
                {isHm && r.f1_saccade != null && (
                    <div className="result-card">
                        <p className="result-label">F1 Saccade</p>
                        <p className="result-value">{r.f1_saccade}</p>
                    </div>
                )}
                <div className="result-card">
                    <p className="result-label">Threshold</p>
                    <p className="result-value">
                        {r.best_threshold ? r.best_threshold.toFixed(2) : 'N/A'}
                    </p>
                    {r.threshold_range && (
                        <p className="result-sublabel">
                            per-sample adaptive: {r.threshold_range.min.toFixed(2)}-{r.threshold_range.max.toFixed(2)}
                        </p>
                    )}
                </div>
            </div>

            {!isHm && plotUrl && (
                <div className="plot-container">
                    <h3>Stationary Visualization</h3>
                    <iframe
                        src={plotUrl}
                        title="Stationary Plot"
                        className="plot-iframe"
                    />
                </div>
            )}

            {!isHm && timePlotUrl && (
                <div className="plot-container">
                    <h3>Time-Scrolling Visualization</h3>
                    <iframe
                        src={timePlotUrl}
                        title="Time-Scrolling Plot"
                        className="plot-iframe"
                    />
                </div>
            )}

            {isHm && videoPlotUrl && (
                <div className="plot-container video-plot-container">
                    <h3>Video Gaze Overlay Visualization</h3>
                    <iframe
                        src={videoPlotUrl}
                        title="Video Gaze Overlay"
                        className="plot-iframe video-iframe"
                        scrolling="no"
                        style={{ height: `${videoFrameHeight}px` }}
                    />
                </div>
            )}
        </div>
    );
}

export default App;
