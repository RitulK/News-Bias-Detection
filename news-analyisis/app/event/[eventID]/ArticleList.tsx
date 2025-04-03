"use client"
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";

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
    <div className="w-full mx-auto p-4 ">
      {/* Always Show Total Articles Count */}
      <h2 className="text-xl font-bold">{articles.length} Articles</h2>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="flex gap-2 mt-3 border border-black rounded-lg bg-white p-1">
          {["All", "Left", "Center", "Right"].map((tab) => (
            <TabsTrigger
              key={tab}
              value={tab}
              className="data-[state=active]:bg-black data-[state=active]:text-white px-4 py-2 rounded-md transition-colors"
            >
              {tab} {tab !== "All" && <span className="text-gray-500">({articles.filter((a) => a.alignment === tab).length})</span>}
            </TabsTrigger>
          ))}
        </TabsList>

        {filteredArticles.map((article) => (
          <TabsContent key={article.id} value={selectedTab} className="mt-4">
            <Card className="border border-black rounded-lg shadow-sm">
              <CardHeader className="flex flex-row items-center space-x-3">
                <img src={article.sourceLogo} alt={article.source} className="h-6 w-6 rounded-full" />
                <span className="text-sm font-semibold">{article.source}</span>
                <span className="ml-auto text-xs px-2 py-1 border border-black rounded-md">{article.alignment}</span>
              </CardHeader>
              <CardContent>
                <CardTitle className="text-lg font-semibold text-black">{article.title}</CardTitle>
                <p className="text-gray-700 mt-2">{article.summary}</p>
                <p className="text-sm text-gray-500 mt-2">{article.timestamp} • {article.location}</p>

                {/* Black "Read Full Article" Button */}
                <Button
                  variant="outline"
                  className="bg-black text-white border border-black hover:bg-gray-900 mt-3 w-full text-sm py-2"
                  asChild
                >
                  <a href={article.link} target="_blank" rel="noopener noreferrer">
                    Read Full Article
                  </a>
                </Button>
              </CardContent>
            </Card>
          </TabsContent>
        ))}
      </Tabs>

      <Button variant="outline" className="w-full mt-4 border-black text-black">
        More articles
      </Button>
    </div>
  );
}
