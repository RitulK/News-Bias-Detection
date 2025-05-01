import { title } from 'process';
import { Suspense } from 'react';
import { json } from 'stream/consumers';

export default async function AnalyzePage({
    searchParams,
}: {
    searchParams: { url?: string };
}) {
    const url = searchParams.url;
    
    if (!url) {
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-gray-100 mt-25">
                <h1>Error: No URL provided</h1>
                <p>Please provide a URL to analyze</p>
            </div>
        );
    }
    
    try {
        const response = await fetch(`http://localhost:8000/analyze/?url=${url}`);
        
        if (!response.ok) {
            throw new Error(`Error fetching analysis: ${response.status}`);
        }
        
        let analysis = await response.json();
        JSON.stringify(analysis.headline);
        // const headline = analysis.headline;

        return (
            <div className="flex flex-col items-center justify-center h-screen bg-gray-100 mt-25">
                <h1>Analyzing URL:{url}</h1>
                <p>{analysis}</p>
                {/* <p>{JSON.stringify(analysis)}</p> */}
            </div>
        );
    } catch (error) {
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-gray-100 mt-25">
                <h1>Error analyzing URL</h1>
                <p>{error instanceof Error ? error.message : 'Unknown error occurred'}</p>
            </div>
        );
    }
}