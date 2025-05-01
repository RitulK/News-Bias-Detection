"use client"
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

interface Article {
  id: number;
  source: string;
  sourceLogo: string;
  title: string;
  summary: string;
  timestamp: string;
  location: string;
  alignment: string;
  link: string;
}

interface ArticlesListProps {
  articles: Article[];
}

export default function ArticlesList({ articles }: ArticlesListProps) {
  const [selectedTab, setSelectedTab] = useState("All");

  const filteredArticles =
    selectedTab === "All"
      ? articles
      : articles.filter((article) => article.alignment === selectedTab);

  return (
    <div className="w-full mx-auto">
      {/* Header with article count */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-full bg-gradient-to-r from-blue-500/10 to-indigo-500/10 border border-blue-100 flex items-center justify-center">
            <span className="text-blue-700 font-medium text-sm">{articles.length}</span>
          </div>
          <h2 className="text-xl font-bold text-slate-800 bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-700">Articles</h2>
        </div>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <div className="relative mb-8">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 to-indigo-500/5 rounded-2xl blur-xl -z-10 transform scale-105"></div>
          <TabsList className="flex gap-1.5 p-1.5 rounded-xl bg-white/80 border border-slate-100 w-fit shadow-md backdrop-blur-sm">
            {["All", "Left", "Center", "Right"].map((tab) => (
              <TabsTrigger
                key={tab}
                value={tab}
                className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-6 py-2 rounded-lg transition-all duration-200 text-sm font-medium relative"
              >
                {tab}
                {tab !== "All" && (
                  <span className={`
                    min-w-[18px] h-[18px] flex items-center justify-center
                    text-[10px] font-bold rounded-full
                    ${selectedTab === tab ? 
                      "bg-white text-blue-600" : 
                      "bg-blue-100 text-blue-600"}
                    border ${selectedTab === tab ? 
                      "border-white/30" : 
                      "border-blue-200"}
                    shadow-sm
                  `}>
                    {articles.filter((a) => a.alignment === tab).length}
                  </span>
                )}
              </TabsTrigger>
            ))}
          </TabsList>
        </div>

        <div className="space-y-4">
          {filteredArticles.length > 0 ? (
            filteredArticles.map((article) => (
              <Card key={article.id} className="border border-slate-200 rounded-xl shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden bg-white/90 backdrop-blur-sm">
                <CardHeader className="flex flex-row items-center space-x-3 py-3 px-5 bg-gradient-to-r from-slate-50 to-white border-b border-slate-100">
                  <div className="flex items-center gap-3">
                    <img 
                      src={article.sourceLogo} 
                      alt={article.source} 
                      className="h-8 w-8 rounded-md object-cover shadow-sm" 
                    />
                    <span className="text-sm font-semibold text-slate-800">{article.source}</span>
                  </div>
                  <span className={`ml-auto text-xs px-3 py-1 rounded-full font-medium ${
                    article.alignment === "Left" ? "bg-blue-50 text-blue-700 border border-blue-100" : 
                    article.alignment === "Center" ? "bg-purple-50 text-purple-700 border border-purple-100" : 
                    "bg-indigo-50 text-indigo-700 border border-indigo-100"
                  }`}>
                    {article.alignment}
                  </span>
                </CardHeader>
                <CardContent className="p-5">
                  <CardTitle className="text-lg font-semibold text-slate-800 mb-3 line-clamp-2">
                    {article.title}
                  </CardTitle>
                  <p className="text-slate-600 text-sm leading-relaxed mb-4">{article.summary}</p>
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-500 flex items-center gap-1.5">
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-slate-400">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                      </svg>
                      {article.timestamp} 
                      <span className="mx-1">•</span>
                      <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-slate-400">
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                        <circle cx="12" cy="10" r="3"></circle>
                      </svg>
                      {article.location}
                    </p>
                    <Button
                      variant="outline"
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-none hover:opacity-90 text-xs rounded-full h-8 px-4 shadow-sm flex items-center gap-1.5"
                      asChild
                    >
                      <a href={article.link} target="_blank" rel="noopener noreferrer">
                        Read Article
                        <ExternalLink size={12} />
                      </a>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))
          ) : (
            <div className="bg-slate-50/80 backdrop-blur-sm p-8 rounded-xl text-center border border-slate-200">
              <p className="text-slate-600">No articles found matching the selected filter.</p>
            </div>
          )}
        </div>
      </Tabs>

      <Button 
        variant="outline" 
        className="w-full mt-6 border border-slate-200 bg-white/80 backdrop-blur-sm text-slate-800 hover:bg-slate-50 transition-colors rounded-xl h-12 shadow-sm"
      >
        <span className="flex items-center gap-2">
          Load More Articles
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 9l6 6 6-6"></path>
          </svg>
        </span>
      </Button>
    </div>
  );
}
