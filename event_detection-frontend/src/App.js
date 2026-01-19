const { useState } = React;

function App() {
    const [file, setFile] = useState(null);
    const [resolution, setResolution] = useState('2560,1440');
    const [minFixationDuration, setMinFixationDuration] = useState(100);
    const [detectThreshold, setDetectThreshold] = useState(125);
    const [algorithm, setAlgorithm] = useState('idt');
    const [samplingRate, setSamplingRate] = useState(250);
    const [fixationMergeThreshold, setFixationMergeThreshold] = useState(null);
    const [adapt, setAdapt] = useState(false);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

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
            formData.append('min_fixation_duration', minFixationDuration.toString());
            formData.append('detect_threshold', detectThreshold.toString());
            formData.append('algorithm', algorithm);
            formData.append('sampling_rate', samplingRate.toString());
            if (fixationMergeThreshold !== null && fixationMergeThreshold !== undefined) {
                formData.append('fixation_merge_threshold', fixationMergeThreshold.toString());
            }
            formData.append('adapt', adapt.toString());

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
                <h1>🔍 Eye Tracking - Fixation Detection</h1>
                <p>Upload gaze data and detect fixations with customizable parameters</p>
            </header>

            <main className="main-content">
                <div className="panel">
                    <FileUpload onFileSelect={handleFileSelect} fileName={file?.name} />
                </div>

                <div className="panel">
                    <h2>⚙️ Detection Parameters</h2>

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
                        <input
                            id="min-fixation"
                            type="number"
                            value={minFixationDuration}
                            onChange={(e) => setMinFixationDuration(parseInt(e.target.value))}
                            className="input-field"
                        />
                    </div>

                    <div className="control-group">
                        <label htmlFor="detect-threshold">Detection Threshold</label>
                        <input
                            id="detect-threshold"
                            type="number"
                            min="0"
                            max="1"
                            step="0.01"
                            value={detectThreshold}
                            onChange={(e) => setDetectThreshold(parseFloat(e.target.value))}
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
                            <option value="idt">IDT (Identification by Two-Thresholds)</option>
                            <option value="ivt">IVT (Identification by Velocity Threshold)</option>
                        </select>
                    </div>

                    <div className="control-group">
                        <label htmlFor="sampling-rate">Sampling Rate (Hz)</label>
                        <input
                            id="sampling-rate"
                            type="number"
                            value={samplingRate}
                            onChange={(e) => setSamplingRate(parseInt(e.target.value))}
                            className="input-field"
                        />
                    </div>

                    <div className="control-group">
                        <label htmlFor="fixation-merge">Fixation Merge Threshold (px, optional)</label>
                        <input
                            id="fixation-merge"
                            type="number"
                            min="0"
                            step="1"
                            value={fixationMergeThreshold || ''}
                            onChange={(e) => setFixationMergeThreshold(e.target.value ? parseFloat(e.target.value) : null)}
                            placeholder="None"
                            className="input-field"
                        />
                    </div>

                    <div className="control-group">
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
                </div>

                <button
                    className="process-button"
                    onClick={handleProcess}
                    disabled={loading || !file}
                >
                    {loading ? 'Processing...' : 'Process Gaze Data'}
                </button>

                {error && <div className="error-message">{error}</div>}

                {loading && <div className="loading-spinner">Processing...</div>}

                {results && <ResultsDisplay results={results} />}
            </main>
        </div>
    );
}

function FileUpload({ onFileSelect, fileName }) {
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

    return (
        <div className="file-upload">
            <h2>📁 Upload Gaze Data</h2>
            <div className="upload-area">
                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    id="csv-input"
                    className="hidden-input"
                />
                <label htmlFor="csv-input" className="upload-label">
                    <div className="upload-icon">📄</div>
                    <p className="upload-text">
                        {fileName ? `Selected: ${fileName}` : 'Click to select CSV file'}
                    </p>
                    <p className="upload-hint">or drag and drop</p>
                </label>
            </div>
            {fileName && <div className="file-selected">✓ File ready for processing</div>}
        </div>
    );
}

function ResultsDisplay({ results }) {
    return (
        <div className="results-container">
            <h2>📊 Detection Results</h2>

            <div className="results-grid">
                <div className="result-card">
                    <div className="result-icon">👁️</div>
                    <div className="result-info">
                        <p className="result-label">Fixations Detected</p>
                        <p className="result-value">
                            {results.result?.num_fixations || 0}
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">⚡</div>
                    <div className="result-info">
                        <p className="result-label">Saccades</p>
                        <p className="result-value">
                            {results.result?.num_saccades || 0}
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">📈</div>
                    <div className="result-info">
                        <p className="result-label">Total Events</p>
                        <p className="result-value">
                            {results.result?.num_events || 0}
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">✓</div>
                    <div className="result-info">
                        <p className="result-label">Status</p>
                        <p className="result-value">{results.success ? 'Complete' : 'Failed'}</p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">🎯</div>
                    <div className="result-info">
                        <p className="result-label">Threshold Used</p>
                        <p className="result-value">
                            {results.result?.best_threshold ? results.result.best_threshold.toFixed(2) : 'N/A'}
                        </p>
                    </div>
                </div>
            </div>

            {results.message && (
                <div className="results-message">
                    <p>{results.message}</p>
                </div>
            )}

            <div className="results-actions">
                {results.result?.events_file && (
                    <a
                        href={`http://127.0.0.1:5000/api/results/${results.filename}`}
                        className="download-button"
                        download
                    >
                        ⬇️ Download Events CSV
                    </a>
                )}
                {results.result?.plot_file && (
                    <button
                        className="view-button"
                        onClick={() => window.open(`http://127.0.0.1:5000/api/plot/${results.filename}`, '_blank')}
                    >
                        📊 View Plot
                    </button>
                )}
            </div>
        </div>
    );
}

// Render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
