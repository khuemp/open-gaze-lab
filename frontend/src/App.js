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
    const [hmSamplingRate, setHmSamplingRate] = useState('');
    const [hmAdapt, setHmAdapt] = useState(false);

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
            appendIfPresent(formData, 'sampling_rate', hmSamplingRate);
            formData.append('adapt', hmAdapt.toString());

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
                                <FileUpload onFileSelect={handleFileSelect} fileName={file?.name} />
                                <BackgroundImageUpload
                                    onImageSelect={handleBackgroundImageSelect}
                                    fileName={backgroundImage?.name}
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
                                <DatasetZipUpload
                                    onFileSelect={setDatasetZip}
                                    fileName={datasetZip?.name}
                                />
                                <VideoUpload
                                    onFileSelect={setVideoFile}
                                    fileName={videoFile?.name}
                                />
                            </div>
                        </div>

                        <div className="panel">
                            <h2>Detection Parameters</h2>
                            <div className="params-grid">
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
                                        placeholder="30"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-min-fixation">Min Fixation Duration (ms)</label>
                                    <NumericInput
                                        id="hm-min-fixation"
                                        value={hmMinFixation}
                                        onChange={setHmMinFixation}
                                        placeholder="50"
                                        className="input-field"
                                    />
                                </div>

                                <div className="control-group">
                                    <label htmlFor="hm-threshold">Detection Threshold</label>
                                    <NumericInput
                                        id="hm-threshold"
                                        value={hmThreshold}
                                        onChange={setHmThreshold}
                                        placeholder="1.0"
                                        className="input-field"
                                    />
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

                {stationaryResults && mode === 'stationary' && <ResultsDisplay results={stationaryResults} />}
                {hmResults && mode === 'headmounted' && <VideoResultsDisplay results={hmResults} />}
            </main>
        </div>
    );
}

function FileUpload({ onFileSelect, fileName }) {
    const [isDragging, setIsDragging] = React.useState(false);

    const handleFileChange = (event) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                onFileSelect(file);
            } else {
                alert('Please select a CSV file');
            }
        }
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

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                onFileSelect(file);
            } else {
                alert('Please select a CSV file');
            }
        }
    };

    return (
        <div className="file-upload">
            <h2>Gaze Data</h2>
            <div
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    id="csv-input"
                    className="hidden-input"
                />
                <label htmlFor="csv-input" className="upload-label">
                    <div className="upload-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" />
                            <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                    <p className="upload-text">
                        {fileName ? `Selected: ${fileName}` : 'Select CSV file'}
                    </p>
                    <p className="upload-hint">or drag and drop</p>
                </label>
            </div>
            {fileName && <div className="file-selected">File ready for processing</div>}
        </div>
    );
}

function BackgroundImageUpload({ onImageSelect, fileName }) {
    const [isDragging, setIsDragging] = React.useState(false);

    const handleFileChange = (event) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                onImageSelect(file);
            } else {
                alert('Please select an image file (PNG, JPG, etc.)');
            }
        }
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

        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                onImageSelect(file);
            } else {
                alert('Please select an image file (PNG, JPG, etc.)');
            }
        }
    };

    const handleClear = (e) => {
        e.preventDefault();
        e.stopPropagation();
        onImageSelect(null);
    };

    return (
        <div className="file-upload background-image-upload">
            <h2>Background Image</h2>
            <div
                className={`upload-area ${isDragging ? 'dragging' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    id="bg-image-input"
                    className="hidden-input"
                />
                <label htmlFor="bg-image-input" className="upload-label">
                    <div className="upload-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <circle cx="8.5" cy="8.5" r="1.5" />
                            <polyline points="21 15 16 10 5 21" />
                        </svg>
                    </div>
                    <p className="upload-text">
                        {fileName ? `Selected: ${fileName}` : 'Select background image'}
                    </p>
                    <p className="upload-hint">or drag and drop</p>
                </label>
            </div>
            {fileName && (
                <div className="file-selected bg-image-selected">
                    Image will be shown behind plots
                    <button onClick={handleClear} className="clear-image-btn">Clear</button>
                </div>
            )}
        </div>
    );
}

function ResultsDisplay({ results }) {
    return (
        <div className="results-container">
            <div className="results-header">
                <h2>Detection Results</h2>
                {results.result?.events_file && (
                    <a
                        href={`http://127.0.0.1:5000/api/results/${results.filename}`}
                        className="download-button"
                        download
                    >
                        Download Events CSV
                    </a>
                )}
            </div>

            <div className="results-grid">
                <div className="result-card">
                    <p className="result-label">Total Gaze Samples</p>
                    <p className="result-value">
                        {results.result?.num_events || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Fixation Samples</p>
                    <p className="result-value">
                        {results.result?.num_fixations || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Saccade Samples</p>
                    <p className="result-value">
                        {results.result?.num_saccades || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Out of Range</p>
                    <p className="result-value">
                        {results.result?.num_oor_gaze_points || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Invalid (NaN)</p>
                    <p className="result-value">
                        {results.result?.num_nan_gaze_points || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Fixation Events</p>
                    <p className="result-value">
                        {results.result?.num_fixation_points || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Threshold</p>
                    <p className="result-value">
                        {results.result?.best_threshold ? results.result.best_threshold.toFixed(2) : 'N/A'}
                    </p>
                </div>
            </div>

            {results.message && (
                <div className="results-message">
                    <p>{results.message}</p>
                </div>
            )}

            {results.result?.plot_file && (
                <div className="plot-container">
                    <h3>Stationary Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot/${results.filename}`}
                        title="Stationary Plot"
                        className="plot-iframe"
                    />
                </div>
            )}

            {results.result?.time_plot_file && (
                <div className="plot-container">
                    <h3>Time-Scrolling Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot-time/${results.filename}`}
                        title="Time-Scrolling Plot"
                        className="plot-iframe"
                    />
                </div>
            )}
        </div>
    );
}

function DatasetZipUpload({ onFileSelect, fileName }) {
    const [isDragging, setIsDragging] = React.useState(false);
    const handleFileChange = (e) => {
        const f = e.target.files?.[0];
        if (f && f.name.toLowerCase().endsWith('.zip')) onFileSelect(f);
        else alert('Please select a .zip file');
    };
    const handleDrop = (e) => {
        e.preventDefault(); setIsDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && f.name.toLowerCase().endsWith('.zip')) onFileSelect(f);
        else alert('Please select a .zip file');
    };
    return (
        <div className="file-upload">
            <h2>Dataset</h2>
            <div className={`upload-area ${isDragging ? 'dragging' : ''}`}
                 onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                 onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                 onDrop={handleDrop}>
                <input type="file" accept=".zip" onChange={handleFileChange}
                       id="zip-input" className="hidden-input" />
                <label htmlFor="zip-input" className="upload-label">
                    <div className="upload-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                    </div>
                    <p className="upload-text">{fileName ? `Selected: ${fileName}` : 'Select ZIP (.npy) dataset'}</p>
                    <p className="upload-hint">gaze, time_gaze, optic_flow, time_optic_flow, time_scene_camera, (gt_labels)</p>
                </label>
            </div>
            {fileName && <div className="file-selected">Dataset ready</div>}
        </div>
    );
}

function VideoUpload({ onFileSelect, fileName }) {
    const [isDragging, setIsDragging] = React.useState(false);
    const handleFileChange = (e) => {
        const f = e.target.files?.[0];
        if (f && f.name.toLowerCase().endsWith('.mp4')) onFileSelect(f);
        else alert('Please select an .mp4 video file');
    };
    const handleDrop = (e) => {
        e.preventDefault(); setIsDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f && f.name.toLowerCase().endsWith('.mp4')) onFileSelect(f);
        else alert('Please select an .mp4 video file');
    };
    return (
        <div className="file-upload">
            <h2>Scene Video</h2>
            <div className={`upload-area ${isDragging ? 'dragging' : ''}`}
                 onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                 onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                 onDrop={handleDrop}>
                <input type="file" accept=".mp4,video/mp4" onChange={handleFileChange}
                       id="video-input" className="hidden-input" />
                <label htmlFor="video-input" className="upload-label">
                    <div className="upload-icon">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
                            <line x1="7" y1="2" x2="7" y2="22" /><line x1="17" y1="2" x2="17" y2="22" />
                            <line x1="2" y1="12" x2="22" y2="12" />
                            <line x1="2" y1="7" x2="7" y2="7" /><line x1="2" y1="17" x2="7" y2="17" />
                            <line x1="17" y1="7" x2="22" y2="7" /><line x1="17" y1="17" x2="22" y2="17" />
                        </svg>
                    </div>
                    <p className="upload-text">{fileName ? `Selected: ${fileName}` : 'Select MP4 video'}</p>
                    <p className="upload-hint">Scene camera recording</p>
                </label>
            </div>
            {fileName && <div className="file-selected">Video ready</div>}
        </div>
    );
}

function VideoResultsDisplay({ results }) {
    const [videoFrameHeight, setVideoFrameHeight] = React.useState(1000);

    React.useEffect(() => {
        const handleMessage = (event) => {
            if (event.data?.type !== 'video-gaze-visualization-height') return;
            const nextHeight = Number(event.data.height);
            if (Number.isFinite(nextHeight) && nextHeight > 0) {
                setVideoFrameHeight((currentHeight) => {
                    const roundedHeight = Math.ceil(nextHeight);
                    return Math.abs(currentHeight - roundedHeight) > 4 ? roundedHeight : currentHeight;
                });
            }
        };

        window.addEventListener('message', handleMessage);
        return () => window.removeEventListener('message', handleMessage);
    }, []);

    return (
        <div className="results-container">
            <div className="results-header">
                <h2>Detection Results</h2>
                {results.result?.events_file && (
                    <a href={`http://127.0.0.1:5000/api/results/${results.filename}`}
                       className="download-button" download>
                        Download Events CSV
                    </a>
                )}
            </div>

            <div className="results-grid hm-results-grid">
                <div className="result-card">
                    <p className="result-label">Total Gaze Samples</p>
                    <p className="result-value">{results.result?.num_events || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Fixation Samples</p>
                    <p className="result-value">{results.result?.num_fixations || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Saccade Samples</p>
                    <p className="result-value">{results.result?.num_saccades || 0}</p>
                </div>
                <div className="result-card">
                    <p className="result-label">Fixation Events</p>
                    <p className="result-value">{results.result?.num_fixation_centers || 0}</p>
                </div>
                {results.result?.f1_fixation != null && (
                    <div className="result-card">
                        <p className="result-label">F1 Fixation</p>
                        <p className="result-value">{results.result.f1_fixation}</p>
                    </div>
                )}
                {results.result?.f1_saccade != null && (
                    <div className="result-card">
                        <p className="result-label">F1 Saccade</p>
                        <p className="result-value">{results.result.f1_saccade}</p>
                    </div>
                )}
                <div className="result-card">
                    <p className="result-label">Threshold</p>
                    <p className="result-value">
                        {results.result?.best_threshold ? results.result.best_threshold.toFixed(2) : 'N/A'}
                    </p>
                </div>
            </div>

            {results.message && (
                <div className="results-message"><p>{results.message}</p></div>
            )}

            {results.result?.video_plot_file && (
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
