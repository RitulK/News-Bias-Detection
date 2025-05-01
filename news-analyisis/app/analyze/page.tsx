"use client"

import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Loader2, Search, ExternalLink, AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import Image from "next/image"

interface AnalysisResult {
  headline: string
  img: string
  text: string
  alignment: string
  Summary: string
}

export default function AnalyzePage() {
  const [url, setUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  const handleAnalyze = async () => {
    if (!url) {
      setError("Please enter a URL to analyze")
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      
      console.log("Analyzing")
          
      // Add timeout to prevent hanging on failed connections
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
      
      try {
        const response = await fetch(`http://localhost:8000/analyze/?url=${encodeURIComponent(url)}`, {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        console.log("Response:", response)

        // if (!response.ok) {
        //   throw new Error(`Analysis failed with status: ${response.status}`);
        // }
        
        const data = await response.json();
        setResult(data);
      } catch (fetchError: any) {
        // Handle specific fetch errors
        if (fetchError.name === 'AbortError') {
          throw new Error('Request timed out. The server might be down or overloaded.');
        } else if (fetchError.message === 'Failed to fetch') {
          throw new Error('Network error: Please check if the backend server is running at http://localhost:8000');
        } else {
          throw fetchError; // Re-throw other errors
        }
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setError(err instanceof Error ? err.message : "Failed to analyze article. Please check your connection and try again.");
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background elements */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 -z-20"></div>
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGZpbGw9InRyYW5zcGFyZW50IiBkPSJNMCAwaDYwdjYwSDB6Ii8+PHBhdGggZD0iTTM2IDM0aC0ydi00aDJ2NHptMC04aC0ydi00aDJ2NHpNMjQgMzBoLTR2Mmg0di0yem04LTRoLTR2Mmg0di0yeiIgZmlsbD0iI2JiYzVlMSIgZmlsbC1ydWxlPSJub256ZXJvIi8+PC9nPjwvc3ZnPg==')] opacity-50 -z-10"></div>
      
      {/* Animated blobs */}
      <div className="fixed top-20 right-20 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob -z-10"></div>
      <div className="fixed bottom-20 left-20 w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000 -z-10"></div>
      
      <div className="container mx-auto pt-24 pb-16 px-6 relative z-10">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center backdrop-blur-md py-8 rounded-3xl bg-white/30 border border-white/50 shadow-xl shadow-blue-900/5 mb-8">
            <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600 mb-3">Article Analysis</h1>
            <div className="h-1 w-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full mx-auto my-4"></div>
            <p className="text-slate-700 max-w-xl mx-auto">Enter the URL of any news article to analyze its political bias and get an AI-generated summary.</p>
          </div>
          
          {/* URL Input Section */}
          <Card className="backdrop-blur-md bg-white/60 border border-white/50 shadow-xl shadow-blue-900/5 rounded-xl mb-8">
            <CardHeader>
              <CardTitle className="text-xl text-slate-800">Enter Article URL</CardTitle>
              <CardDescription>Paste a URL to any news article you want to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <div className="relative flex-grow">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400" size={18} />
                  <Input
                    placeholder="https://example.com/article"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="pl-10 bg-white/80 border-slate-200"
                  />
                </div>
                <Button 
                  onClick={handleAnalyze} 
                  disabled={isLoading || !url}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white hover:opacity-90"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing
                    </>
                  ) : "Analyze"}
                </Button>
              </div>
            </CardContent>
            {error && (
              <CardFooter>
                <Alert variant="destructive" className="w-full">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>
                    {error}
                  </AlertDescription>
                </Alert>
              </CardFooter>
            )}
          </Card>
          
          {/* Results Section */}
          {result && (
            <div className="space-y-8">
              <Card className="backdrop-blur-md bg-white/60 border border-white/50 shadow-xl shadow-blue-900/5 rounded-xl overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-indigo-600 h-3 w-full"></div>
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-2xl font-bold text-slate-800">{result.headline}</CardTitle>
                      <CardDescription className="mt-2">
                        <div className="flex items-center gap-2">
                          <span>Original URL:</span>
                          <a 
                            href={url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 hover:underline flex items-center gap-1"
                          >
                            {url.length > 50 ? url.substring(0, 50) + '...' : url}
                            <ExternalLink size={14} />
                          </a>
                        </div>
                      </CardDescription>
                    </div>
                    <Badge 
                      className={`
                        px-3 py-1 text-sm font-medium shadow-sm
                        ${result.alignment.toLowerCase() === 'left' ? 'bg-blue-100 text-blue-700 border border-blue-200' : 
                          result.alignment.toLowerCase() === 'center' ? 'bg-purple-100 text-purple-700 border border-purple-200' : 
                          'bg-indigo-100 text-indigo-700 border border-indigo-200'}
                      `}
                    >
                      {result.alignment} Leaning
                    </Badge>
                  </div>
                </CardHeader>
                
                <CardContent>
                  <Tabs defaultValue="analysis" className="w-full">
                    <TabsList className="flex gap-1.5 p-1.5 rounded-xl bg-white/80 border border-slate-100 w-fit shadow-md backdrop-blur-sm mb-6">
                      <TabsTrigger
                        value="analysis"
                        className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-6 py-2 rounded-lg transition-all duration-200 text-sm font-medium"
                      >
                        Analysis
                      </TabsTrigger>
                      <TabsTrigger
                        value="summary"
                        className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-6 py-2 rounded-lg transition-all duration-200 text-sm font-medium"
                      >
                        Summary
                      </TabsTrigger>
                      <TabsTrigger
                        value="full-text"
                        className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-6 py-2 rounded-lg transition-all duration-200 text-sm font-medium"
                      >
                        Full Text
                      </TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="analysis" className="space-y-6">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Visualization */}
                        <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-100 p-5 shadow-sm">
                          <h3 className="text-lg font-semibold text-slate-800 mb-4">Political Bias Analysis</h3>
                          
                          {/* Bias Gauge */}
                          <div className="relative h-8 w-full bg-slate-100 rounded-full mb-6">
                            <div className="absolute inset-0 flex">
                              <div className="w-1/3 bg-blue-500 rounded-l-full opacity-25"></div>
                              <div className="w-1/3 bg-purple-500 opacity-25"></div>
                              <div className="w-1/3 bg-indigo-500 rounded-r-full opacity-25"></div>
                            </div>
                            
                            {/* Indicator */}
                            <div 
                              className="absolute top-0 h-8 w-8 bg-white rounded-full border-4 border-blue-600 shadow-lg transform -translate-x-1/2"
                              style={{ 
                                left: result.alignment.toLowerCase() === 'left' ? '16.67%' : 
                                      result.alignment.toLowerCase() === 'center' ? '50%' : '83.33%' 
                              }}
                            ></div>
                            
                            {/* Labels */}
                            <div className="absolute -bottom-6 left-0 text-xs text-blue-700 font-medium">Left</div>
                            <div className="absolute -bottom-6 left-1/2 transform -translate-x-1/2 text-xs text-purple-700 font-medium">Center</div>
                            <div className="absolute -bottom-6 right-0 text-xs text-indigo-700 font-medium">Right</div>
                          </div>
                          
                          {/* Analysis Details */}
                          <div className="mt-10">
                            <div className="flex items-center justify-between mb-3">
                              <span className="text-sm font-medium text-slate-600">Bias Confidence:</span>
                              <div className="h-2 w-2/3 bg-slate-200 rounded-full overflow-hidden">
                                <div className="h-full bg-blue-500" style={{ width: '75%' }}></div>
                              </div>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-slate-600">Source Reliability:</span>
                              <div className="h-2 w-2/3 bg-slate-200 rounded-full overflow-hidden">
                                <div className="h-full bg-green-500" style={{ width: '85%' }}></div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        {/* Article Featre Image */}
                        {result.img && (
                          <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-100 p-5 shadow-sm">
                            <h3 className="text-lg font-semibold text-slate-800 mb-4">Featured Image</h3>
                            <div className="w-full aspect-video relative rounded-lg overflow-hidden">
                              <img 
                                src={result.img} 
                                alt={result.headline}
                                className="object-cover w-full h-full"
                              />
                            </div>
                          </div>
                        )}
                      </div>
                      
                      {/* Language Analysis */}
                      <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-100 p-5 shadow-sm">
                        <h3 className="text-lg font-semibold text-slate-800 mb-4">Language Analysis</h3>
                        <div className="space-y-4">
                          <div>
                            <p className="text-sm mb-1 font-medium text-slate-700">Emotional Language</p>
                            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                              <div className="h-full bg-gradient-to-r from-blue-400 to-blue-600" style={{ width: result.alignment.toLowerCase() === 'left' ? '70%' : result.alignment.toLowerCase() === 'center' ? '40%' : '60%' }}></div>
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-sm mb-1 font-medium text-slate-700">Partisan Framing</p>
                            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                              <div className="h-full bg-gradient-to-r from-indigo-400 to-indigo-600" style={{ width: result.alignment.toLowerCase() === 'left' ? '65%' : result.alignment.toLowerCase() === 'center' ? '35%' : '75%' }}></div>
                            </div>
                          </div>
                          
                          <div>
                            <p className="text-sm mb-1 font-medium text-slate-700">Neutral Presentation</p>
                            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                              <div className="h-full bg-gradient-to-r from-green-400 to-green-600" style={{ width: result.alignment.toLowerCase() === 'left' ? '35%' : result.alignment.toLowerCase() === 'center' ? '75%' : '30%' }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="summary" className="space-y-6">
                      <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-100 p-5 shadow-sm">
                        <h3 className="text-lg font-semibold text-slate-800 mb-4">AI-Generated Summary</h3>
                        <p className="text-slate-700 leading-relaxed">{result.Summary}</p>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="full-text" className="space-y-6">
                      <div className="bg-gradient-to-br from-slate-50 to-white rounded-xl border border-slate-100 p-5 shadow-sm">
                        <h3 className="text-lg font-semibold text-slate-800 mb-4">Full Article Text</h3>
                        <div className="max-h-96 overflow-y-auto pr-4 text-slate-700">
                          <p className="whitespace-pre-line leading-relaxed">{result.text}</p>
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
                
                <CardFooter className="bg-slate-50/50 border-t border-slate-100 text-xs text-slate-500">
                  <div className="flex items-center gap-1.5">
                    <AlertCircle size={14} />
                    AI-powered analysis. Results may not be perfectly accurate.
                  </div>
                </CardFooter>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
