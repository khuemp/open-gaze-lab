const { useState } = React;

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

function App() {
    const appendIfPresent = (formData, key, value) => {
        if (value !== '') {
            formData.append(key, value);
        }
    };

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

    // Shared state
    const [stationaryResults, setStationaryResults] = useState(null);
    const [hmResults, setHmResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleBackgroundImageSelect = (selectedFile) => {
        setBackgroundImage(selectedFile);
    };

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setError(null);
    };

    const handleModeSwitch = (newMode) => {
        setMode(newMode);
        setError(null);
    };

    const handleProcessHeadMounted = async () => {
        setLoading(true);
        setError(null);
        setHmResults(null);

        try {
            const formData = new FormData();
            formData.append('dataset_zip', datasetZip);
            formData.append('video', videoFile);
            appendIfPresent(formData, 'resolution', hmResolution);
            appendIfPresent(formData, 'min_fixation_duration', hmMinFixation);
            appendIfPresent(formData, 'detection_threshold', hmThreshold);
            appendIfPresent(formData, 'algorithm', hmAlgorithm);
            appendIfPresent(formData, 'sampling_rate', hmSamplingRate);
            formData.append('adapt', hmAdapt.toString());
            appendIfPresent(formData, 'gain', hmGain);
            appendIfPresent(formData, 'window_size_ms', hmWindowSizeMs);

            const response = await fetch('http://127.0.0.1:5000/api/upload-video', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `API error: ${response.statusText}`);
            }

            const data = await response.json();
            setHmResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error processing video data');
        } finally {
            setLoading(false);
        }
    };

    const handleProcessStationary = async () => {
        if (!file) {
            setError('Please select a CSV file first');
            return;
        }

        setLoading(true);
        setError(null);
        setStationaryResults(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            appendIfPresent(formData, 'resolution', resolution);
            appendIfPresent(formData, 'min_fixation_duration', minFixationDuration);
            appendIfPresent(formData, 'detection_threshold', detectionThreshold);
            appendIfPresent(formData, 'algorithm', algorithm);
            appendIfPresent(formData, 'sampling_rate', samplingRate);
            appendIfPresent(formData, 'fixation_merge_threshold', fixationMergeThreshold);
            formData.append('adapt', adapt.toString());
            appendIfPresent(formData, 'y_origin', yOrigin);
            if (backgroundImage) {
                formData.append('background_image', backgroundImage);
            }

            const response = await fetch('http://127.0.0.1:5000/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || errorData.error || `API error: ${response.statusText}`);
            }

            const data = await response.json();
            setStationaryResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error processing gaze data');
        } finally {
            setLoading(false);
        }
    };

    const handleProcess = async () => {
        if (mode === 'headmounted') {
            return handleProcessHeadMounted();
        }

        return handleProcessStationary();
    };

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
                                        placeholder="125"
                                        className="input-field"
                                    />
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
                                    onFileSelect={setVideoFile}
                                    placeholderText="Select MP4 video"
                                    hintText="Scene camera recording"
                                    selectedText="Video ready"
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
                                        placeholder="30"
                                        className="input-field"
                                    />
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
                                disabled={loading || !datasetZip || !videoFile}
                            >
                                {loading ? 'Processing...' : 'Process Gaze Data'}
                            </button>
                        </div>
                    </React.Fragment>
                )}

                {error && <div className="error-message">{error}</div>}

                {loading && <div className="loading-spinner">Processing...</div>}

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
    const [isDragging, setIsDragging] = React.useState(false);

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

function ResultsDisplay({ results, variant = 'stationary' }) {
    const r = results.result || {};
    const isHm = variant === 'headmounted';
    const [videoFrameHeight, setVideoFrameHeight] = React.useState(1000);

    React.useEffect(() => {
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
                {r.events_file && (
                    <a
                        href={`http://127.0.0.1:5000/api/results/${results.filename}`}
                        className="download-button"
                        download
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

            {results.message && (
                <div className="results-message"><p>{results.message}</p></div>
            )}

            {!isHm && r.plot_file && (
                <div className="plot-container">
                    <h3>Stationary Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot/${results.filename}`}
                        title="Stationary Plot"
                        className="plot-iframe"
                    />
                </div>
            )}

            {!isHm && r.time_plot_file && (
                <div className="plot-container">
                    <h3>Time-Scrolling Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot-time/${results.filename}`}
                        title="Time-Scrolling Plot"
                        className="plot-iframe"
                    />
                </div>
            )}

            {isHm && r.video_plot_file && (
                <div className="plot-container video-plot-container">
                    <h3>Video Gaze Overlay Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot-video/${results.filename}`}
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

// Render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
