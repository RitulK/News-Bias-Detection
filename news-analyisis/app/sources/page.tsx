import { connectToDatabase } from '@/lib/mongodb';
import { NewsSource } from '@/types/news';

export default async function NewsSourcesPage() {
  try {
    const { db } = await connectToDatabase();
    const rawSources = await db.collection('news_source1').find().toArray();
    
    const sources: NewsSource[] = rawSources.map(doc => ({
      ...doc,
      _id: doc._id.toString(),
      source_name: doc.source_name,
      alignment_counts: doc.alignment_counts || { Left: 0, Center: 0, Right: 0 },
      last_updated: doc.last_updated,
      total_articles: doc.total_articles || 0,
      source_id: doc.source_id
    }));

    console.log('Processed sources:', sources);

    return (
      <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-3xl font-bold text-gray-900 sm:text-4xl">News Sources Analysis</h1>
            <p className="mt-3 text-xl text-gray-500">
              Detailed breakdown of political alignment across news sources
            </p>
            <p className="mt-2 text-sm text-blue-600">
              Showing {sources.length} sources
            </p>
          </div>

          {sources.length > 0 ? (
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {sources.map((source) => (
                            <div
                            key={source._id.toString()}
                            className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-300"
                          >
                            <div className="p-6">
                              <div className="flex justify-between items-start">
                                <h2 className="text-xl font-semibold text-gray-800 mb-2">{source.source_name}</h2>
                                <span className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full">
                                  {source.total_articles} articles
                                </span>
                              </div>
              
                              <div className="mt-4">
                                <h3 className="text-sm font-medium text-gray-500 mb-2">Political Alignment</h3>
                                <div className="space-y-2">
                                  <div className="flex justify-between">
                                    <span className="text-sm font-medium text-red-600">Left</span>
                                    <span className="text-sm text-gray-700">
                                      {source.alignment_counts.Left} (
                                      {Math.round((source.alignment_counts.Left / source.total_articles) * 100)}%)
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-sm font-medium text-yellow-600">Center</span>
                                    <span className="text-sm text-gray-700">
                                      {source.alignment_counts.Center} (
                                      {Math.round((source.alignment_counts.Center / source.total_articles) * 100)}%)
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-sm font-medium text-blue-600">Right</span>
                                    <span className="text-sm text-gray-700">
                                      {source.alignment_counts.Right} (
                                      {Math.round((source.alignment_counts.Right / source.total_articles) * 100)}%)
                                    </span>
                                  </div>
                                </div>
                              </div>
              
                              <div className="mt-6 pt-4 border-t border-gray-200">
                                <div className="flex justify-between text-sm text-gray-500">
                                  <span>Last updated</span>
                                  <span>
                                    {new Date(source.last_updated).toLocaleDateString('en-US', {
                                      year: 'numeric',
                                      month: 'short',
                                      day: 'numeric',
                                      hour: '2-digit',
                                      minute: '2-digit',
                                    })}
                                  </span>
                                </div>
                                {source.source_id && (
                                  <div className="mt-1 flex justify-between text-sm text-gray-500">
                                    <span>Source ID</span>
                                    <span className="truncate max-w-[120px]">{source.source_id}</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-10">
              <p className="text-lg text-gray-600">No news sources found in database</p>
            </div>
          )}
        </div>
      </div>
    );
  } catch (error) {
    console.error('Database connection failed:', error);
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center text-red-600">
          <h1 className="text-2xl font-bold">Connection Error</h1>
          <p className="mt-4">
            {error instanceof Error ? error.message : 'Failed to connect to database'}
          </p>
          <p className="mt-2 text-sm text-gray-600">
            Check your MongoDB Atlas connection settings
          </p>
        </div>
      </div>
    );
  }
}