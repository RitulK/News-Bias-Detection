import React from 'react';

const MethodologyPage = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-bold text-blue-800 mb-4">
          News Bias Detection Methodology
        </h1>
        <p className="text-xl text-gray-600">
          How we analyze and categorize news articles using Deep learning
        </p>
      </header>

      <section className="mb-16">
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold text-blue-700 mb-4 flex items-center">
            <span className="mr-2">📰</span>
            News Bias Detection & Semantic Similarity Analysis
          </h2>
          <p className="text-gray-700 mb-4">
            This project automatically detects political bias in news articles and groups similar 
            articles covering the same events using fine-tuned transformer models.
          </p>
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-medium text-blue-800 mb-2">Core ML Modules:</h3>
            <ul className="list-disc pl-5 space-y-1">
              <li>MPNET-based Similarity Detection Model</li>
              <li>LLaMA 3-based Bias Detection Model</li>
              <li>Fine-Tuned BERT Model for Bias Classification Enhancement</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="mb-16">
        <h2 className="text-3xl font-semibold text-blue-700 mb-6 flex items-center">
          <span className="mr-2">📌</span>
          Key Features
        </h2>
        <div className="grid md:grid-cols-2 gap-6">
          {[
            "🔍 Semantic Similarity Model (MPNET)",
            "🏛️ Political Bias Detection Models (LLaMA 3.2B & fine-tuned BERT)",
            "🌐 Web Interface using Next.js",
            "🚀 FastAPI-based API for model and DB communication",
            "🗃️ MongoDB database for storing articles and metadata",
            "📊 Confusion Matrices & Evaluation Stats for all models",
            "⚙️ End-to-End pipeline for real-time bias analysis",
            "🧠 Advanced clustering algorithms for event grouping"
          ].map((feature, index) => (
            <div key={index} className="bg-white p-4 rounded-lg shadow-sm border-l-4 border-blue-500">
              <p className="text-gray-800">{feature}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="text-3xl font-semibold text-blue-700 mb-6 flex items-center">
          <span className="mr-2">🧠</span>
          Technology Stack
        </h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded-lg overflow-hidden">
            <thead className="bg-blue-600 text-white">
              <tr>
                <th className="py-3 px-4 text-left">Layer</th>
                <th className="py-3 px-4 text-left">Tech Stack</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {[
                ["Frontend", "Next.js, TailwindCSS"],
                ["Backend API", "FastAPI"],
                ["Database", "MongoDB, pymongo"],
                ["ML Models", "sentence-transformers, transformers, datasets, torch, BERT, LLaMA"],
                ["Training", "MPNET + STSB Dataset (stsb_multi_mt)"],
                ["Bias Models", "LLaMA 3.2B + Fine-Tuned BERT + custom political dataset"],
                
              ].map(([layer, tech], index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-3 px-4 font-medium">{layer}</td>
                  <td className="py-3 px-4">{tech}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-12 bg-blue-50 p-6 rounded-lg">
        <h2 className="text-2xl font-semibold text-blue-700 mb-4">Methodology Overview</h2>
        <div className="space-y-4">
          <p className="text-gray-700">
            Our pipeline begins with article collection and preprocessing, where we extract key 
            textual features and metadata. The content then passes through our similarity detection 
            model to identify related articles.
          </p>
          <p className="text-gray-700">
            For bias analysis, articles are processed through both our LLaMA-based model for 
            broad-stroke classification and our fine-tuned BERT model for nuanced bias detection. 
            Results are stored in MongoDB with detailed metadata for further analysis.
          </p>
          <p className="text-gray-700">
            The frontend interface visualizes these relationships and biases through interactive 
            dashboards and clustered representations.
          </p>
        </div>
      </section>
    </div>
  );
};

export default MethodologyPage;