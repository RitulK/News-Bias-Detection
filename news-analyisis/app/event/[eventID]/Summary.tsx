"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

interface Event {
  // publishedDate: string;
  location: string;
  updateTime: string;
  eventHeadline: string;
  leftSummary: string;
  centerSummary: string;
  rightSummary: string;
}

interface SummaryProps {
  event: Event;
}

export default function Summary({ event }: SummaryProps) {
  const { /*publishedDate,*/ location, updateTime, eventHeadline, leftSummary, centerSummary, rightSummary } = event;

  return (
    <div className="w-full mx-auto p-6">
      <div className="mb-4 flex items-center gap-2 text-sm font-medium">
        <div className="flex items-center gap-1.5 text-slate-500">
          {/* <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
            <path d="M12 20v-6m0 0V4m0 10h6m-6 0H6"></path>
          </svg> */}
          <span>Updated {updateTime}</span>
        </div>
        <span className="text-slate-400">•</span>
        <div className="flex items-center gap-1.5">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500">
            <path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"></path>
            <circle cx="12" cy="10" r="3"></circle>
          </svg>
          <span className="text-blue-600 hover:text-blue-700 cursor-pointer font-medium">{location}</span>
        </div>
      </div>

      <div className="mb-6">
        <h1 className="text-3xl font-bold text-slate-800 leading-tight">{eventHeadline}</h1>
        <div className="h-1 w-16 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full mt-4"></div>
      </div>

      <div className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-100 shadow-lg hover:shadow-xl transition-shadow duration-300 overflow-hidden">
        <div className="p-5 pb-0">
          <Tabs defaultValue="left" className="w-full">
            <TabsList className="flex justify-start p-1 rounded-xl bg-gradient-to-r from-slate-50 to-slate-100/80 border border-slate-200/80 w-fit h-10 ">
              <TabsTrigger
                value="left"
                className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-5 py-1.5 rounded-lg transition-all duration-300 text-sm font-medium"
              >
                Left Perspective
              </TabsTrigger>
              <TabsTrigger
                value="center"
                className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-5 py-1.5 rounded-lg transition-all duration-300 text-sm font-medium"
              >
                Center Perspective
              </TabsTrigger>
              <TabsTrigger
                value="right"
                className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-md px-5 py-1.5 rounded-lg transition-all duration-300 text-sm font-medium"
              >
                Right Perspective
              </TabsTrigger>
            </TabsList>

            {["left", "center", "right"].map((tab, index) => (
              <TabsContent key={tab} value={tab} className="p-1 pt-2">
                <div className="p-5 rounded-lg bg-gradient-to-br from-slate-50 to-white border border-slate-100 transition-all duration-300 hover:shadow-md">
                  <div className="flex items-center gap-2 mb-3">
                    <div className={`w-2 h-2 ${
                      tab === "left" ? "bg-blue-500" : 
                      tab === "center" ? "bg-purple-500" : 
                      "bg-indigo-500"
                    } rounded-full ${
                      tab === "left" ? "animate-pulse" : ""
                    }`}></div>
                    <h3 className={`text-sm font-medium ${
                      tab === "left" ? "text-blue-700" : 
                      tab === "center" ? "text-purple-700" : 
                      "text-indigo-700"
                    }`}>
                      {tab === "left" ? "Left" : tab === "center" ? "Center" : "Right"}-Leaning Perspective
                    </h3>
                  </div>
                  <p className="text-slate-700 leading-relaxed">
                    {tab === "left" ? leftSummary : 
                     tab === "center" ? centerSummary : 
                     rightSummary}
                  </p>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </div>
        
        <div className="px-5 py-3 mt-4 border-t border-slate-100 flex justify-between items-center bg-gradient-to-r from-slate-50/50 to-white/50">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            {/* <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z"></path>
              <path d="m15 9-6 6"></path>
              <path d="m9 9 6 6"></path>
            </svg> */}

        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="12" y1="8" x2="12" y2="12"></line>
                      <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
            <span>AI-generated summary. May contain inaccuracies.</span>
          </div>
          <div className="text-xs bg-gradient-to-r from-blue-50 to-indigo-50 text-blue-600 px-3 py-1 rounded-full font-medium border border-blue-100/50">
            Multi-perspective
          </div>
        </div>
      </div>
    </div>
  );
}
