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

    const handlePaste = (e) => {
        e.preventDefault();
        const pastedText = e.clipboardData.getData('text');
        let pattern = allowDecimal ? /[^\d.]/g : /[^\d]/g;
        if (allowNegative) pattern = allowDecimal ? /[^\d.-]/g : /[^\d-]/g;
        const cleanedText = pastedText.replace(pattern, '');
        const newValue = value.substring(0, e.target.selectionStart) + cleanedText + value.substring(e.target.selectionEnd);
        onChange(newValue);
    };

    const handleChange = (e) => {
        onChange(e.target.value);
    };

    return (
        <input
            id={id}
            type="text"
            inputMode="decimal"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={placeholder}
            className={className}
        />
    );
}

function App() {
    const [file, setFile] = useState(null);
    const [backgroundImage, setBackgroundImage] = useState(null);
    const [resolution, setResolution] = useState('2560,1440');
    const [minFixationDuration, setMinFixationDuration] = useState('100');
    const [detectThreshold, setDetectThreshold] = useState('125');
    const [algorithm, setAlgorithm] = useState('idt');
    const [samplingRate, setSamplingRate] = useState('250');
    const [fixationMergeThreshold, setFixationMergeThreshold] = useState('');
    const [adapt, setAdapt] = useState(false);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleBackgroundImageSelect = (selectedFile) => {
        setBackgroundImage(selectedFile);
    };

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setError(null);
    };

    const handleProcess = async () => {
        if (!file) {
            setError('Please select a CSV file first');
            return;
        }

        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('resolution', resolution);
            formData.append('min_fixation_duration', minFixationDuration);
            formData.append('detect_threshold', detectThreshold);
            formData.append('algorithm', algorithm);
            formData.append('sampling_rate', samplingRate);
            if (fixationMergeThreshold) {
                formData.append('fixation_merge_threshold', fixationMergeThreshold);
            }
            formData.append('adapt', adapt.toString());
            if (backgroundImage) {
                formData.append('background_image', backgroundImage);
            }

            const response = await fetch('http://127.0.0.1:5000/api/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `API error: ${response.statusText}`);
            }

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container">
            <header className="header">
                <h1>Eye-tracking Event Detection</h1>
                <p>Upload gaze data and detect events with customizable parameters</p>
            </header>

            <main className="main-content">
                <div className="panel">
                    <div className="upload-row">
                        <FileUpload onFileSelect={handleFileSelect} fileName={file?.name} />
                        <BackgroundImageUpload
                            onImageSelect={handleBackgroundImageSelect}
                            fileName={backgroundImage?.name}
                        />
                    </div>
                    {results?.result?.events_file && (
                        <a
                            href={`http://127.0.0.1:5000/api/results/${results.filename}`}
                            className="download-button"
                            download
                            style={{ marginTop: '0.75rem', display: 'inline-block' }}
                        >
                            Download Events CSV
                        </a>
                    )}
                </div>

                <div className="panel">
                    <h2>Detection Parameters</h2>
                    <div className="params-grid">
                        <div className="control-group">
                            <label htmlFor="resolution">Display Resolution (WxH)</label>
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
                            <label htmlFor="min-fixation">Min Fixation Duration (ms)</label>
                            <NumericInput
                                id="min-fixation"
                                value={minFixationDuration}
                                onChange={setMinFixationDuration}
                                className="input-field"
                            />
                        </div>

                        <div className="control-group">
                            <label htmlFor="detect-threshold">Detection Threshold</label>
                            <NumericInput
                                id="detect-threshold"
                                value={detectThreshold}
                                onChange={setDetectThreshold}
                                className="input-field"
                            />
                        </div>

                        <div className="control-group">
                            <label htmlFor="algorithm">Algorithm</label>
                            <select
                                id="algorithm"
                                value={algorithm}
                                onChange={(e) => setAlgorithm(e.target.value)}
                                className="input-field"
                            >
                                <option value="idt">IDT</option>
                                <option value="ivt">IVT</option>
                            </select>
                        </div>

                        <div className="control-group">
                            <label htmlFor="sampling-rate">Sampling Rate (Hz)</label>
                            <NumericInput
                                id="sampling-rate"
                                value={samplingRate}
                                onChange={setSamplingRate}
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

                {error && <div className="error-message">{error}</div>}

                {loading && <div className="loading-spinner">Processing...</div>}

                {results && <ResultsDisplay results={results} />}
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
            <h2>Upload Gaze Data</h2>
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
            <h2>Detection Results</h2>

            <div className="results-grid">
                <div className="result-card">
                    <p className="result-label">Total Events</p>
                    <p className="result-value">
                        {results.result?.num_events || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Gaze Points As Fixation</p>
                    <p className="result-value">
                        {results.result?.num_fixations || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Gaze Points As Saccade</p>
                    <p className="result-value">
                        {results.result?.num_saccades || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Gaze Points As Blink</p>
                    <p className="result-value">
                        {results.result?.num_blinks || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Fixation Points</p>
                    <p className="result-value">
                        {results.result?.num_fixation_points || 0}
                    </p>
                </div>

                <div className="result-card">
                    <p className="result-label">Threshold Used</p>
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
                    <h3>Static Visualization</h3>
                    <iframe
                        src={`http://127.0.0.1:5000/api/plot/${results.filename}`}
                        title="Static Plot"
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

// Render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
