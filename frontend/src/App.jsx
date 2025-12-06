import React, { useState } from 'react';
import VideoUpload from './components/VideoUpload';
import Results from './components/Results';
import axios from 'axios';

function App() {
    const [results, setResults] = useState(null);

    const handleAnalyze = async (file) => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Assuming backend is running on localhost:8000
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResults(response.data);
        } catch (error) {
            console.error("Error analyzing video:", error);
            alert("Error analyzing video. Ensure backend is running.");
        }
    };

    return (
        <div className="dark min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-black to-black p-8 font-sans antialiased text-white">
            <div className="container mx-auto max-w-7xl space-y-12">
                <div className="text-center space-y-4 py-8">
                    <h1 className="text-5xl font-extrabold tracking-tight lg:text-6xl bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
                        Action Recognition AI
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                        Advanced human action classification using 3D CNNs and Skeleton Extraction.
                        Upload a video to see the AI in action.
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                    <div className="space-y-6">
                        <div className="flex items-center space-x-2">
                            <div className="h-8 w-1 bg-primary rounded-full" />
                            <h2 className="text-2xl font-semibold tracking-tight">Input Video</h2>
                        </div>
                        <VideoUpload onAnalyze={handleAnalyze} />
                    </div>

                    <div className="space-y-6">
                        <div className="flex items-center space-x-2">
                            <div className="h-8 w-1 bg-blue-600 rounded-full" />
                            <h2 className="text-2xl font-semibold tracking-tight">Analysis Results</h2>
                        </div>
                        {results ? (
                            <Results results={results} />
                        ) : (
                            <div className="h-full min-h-[400px] flex items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/50">
                                <div className="text-center space-y-2 text-muted-foreground">
                                    <p>No analysis yet.</p>
                                    <p className="text-sm">Upload a video to see results here.</p>
                                </div>
                            </div>
                        )}

                        {/* Model Performance Section */}
                        <div className="mt-8 pt-8 border-t border-gray-800">
                            <div className="flex items-center space-x-2 mb-6">
                                <div className="h-8 w-1 bg-green-500 rounded-full" />
                                <h2 className="text-2xl font-semibold tracking-tight">Model Performance</h2>
                            </div>
                            <div className="bg-card rounded-lg border shadow-sm p-6">
                                <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
                                <div className="flex justify-center">
                                    <img
                                        src="http://localhost:8000/static/confusion_matrix.png"
                                        alt="Confusion Matrix"
                                        className="max-w-full h-auto rounded-lg border"
                                        onError={(e) => {
                                            e.target.style.display = 'none';
                                            e.target.parentNode.innerHTML = '<p class="text-muted-foreground">Confusion matrix not available yet.</p>';
                                        }}
                                    />
                                </div>
                                <p className="text-sm text-muted-foreground mt-4 text-center">
                                    Visualizing classification performance on the validation dataset.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;
