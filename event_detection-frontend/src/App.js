const { useState } = React;

function App() {
    const [file, setFile] = useState(null);
    const [threshold, setThreshold] = useState(0.5);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileSelect = (selectedFile) => {
        setFile(selectedFile);
        setError(null);
    };

    const handleThresholdChange = (value) => {
        setThreshold(value);
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
            formData.append('threshold', threshold.toString());

            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.statusText}`);
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
                    <ThresholdControls
                        threshold={threshold}
                        onChange={handleThresholdChange}
                    />
                </div>

                <button
                    className="process-button"
                    onClick={handleProcess}
                    disabled={loading || !file}
                >
                    {loading ? 'Processing...' : 'Process Gaze Data'}
                </button>

                {error && <div className="error-message">{error}</div>}

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

function ThresholdControls({ threshold, onChange }) {
    return (
        <div className="threshold-controls">
            <h2>⚙️ Detection Parameters</h2>

            <div className="control-group">
                <label htmlFor="threshold-slider">Fixation Threshold</label>
                <div className="slider-container">
                    <input
                        id="threshold-slider"
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={threshold}
                        onChange={(e) => onChange(parseFloat(e.target.value))}
                        className="slider"
                    />
                    <span className="threshold-value">{threshold.toFixed(2)}</span>
                </div>
                <p className="threshold-hint">
                    Lower values: more sensitive (detects more fixations)
                    <br />
                    Higher values: more strict (detects fewer fixations)
                </p>
            </div>

            <div className="presets">
                <p className="presets-label">Quick presets:</p>
                <div className="preset-buttons">
                    <button
                        className={`preset-btn ${threshold === 0.3 ? 'active' : ''}`}
                        onClick={() => onChange(0.3)}
                    >
                        Sensitive
                    </button>
                    <button
                        className={`preset-btn ${threshold === 0.5 ? 'active' : ''}`}
                        onClick={() => onChange(0.5)}
                    >
                        Medium
                    </button>
                    <button
                        className={`preset-btn ${threshold === 0.8 ? 'active' : ''}`}
                        onClick={() => onChange(0.8)}
                    >
                        Strict
                    </button>
                </div>
            </div>
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
                            {results.fixation_count || 0}
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">⏱️</div>
                    <div className="result-info">
                        <p className="result-label">Processing Time</p>
                        <p className="result-value">
                            {results.processing_time?.toFixed(2) || '0'}s
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">📈</div>
                    <div className="result-info">
                        <p className="result-label">Data Points</p>
                        <p className="result-value">
                            {results.total_points || 0}
                        </p>
                    </div>
                </div>

                <div className="result-card">
                    <div className="result-icon">✓</div>
                    <div className="result-info">
                        <p className="result-label">Status</p>
                        <p className="result-value">Complete</p>
                    </div>
                </div>
            </div>

            {results.message && (
                <div className="results-message">
                    <p>{results.message}</p>
                </div>
            )}

            {results.details && (
                <div className="results-details">
                    <h3>Additional Information</h3>
                    <pre>{JSON.stringify(results.details, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}

// Render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
