"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface CoverageProps {
  totalSources: number;
  leaningLeft: number;
  leaningRight: number;
  center: number;
  lastUpdated: string;
  biasDistribution: string;
}

export default function CoverageDetails({
  totalSources,
  leaningLeft,
  leaningRight,
  center,
  lastUpdated,
  biasDistribution,
}: CoverageProps) {
  // Calculate percentages for proper chart rendering
  const leftPercent = Math.round((leaningLeft/totalSources)*100);
  const centerPercent = Math.round((center/totalSources)*100);
  const rightPercent = Math.round((leaningRight/totalSources)*100);

  return (
    <Card className="bg-white/80 backdrop-blur-sm rounded-xl border border-slate-100 shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 h-3 w-full"></div>
      <CardHeader className="pb-2 pt-5">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold text-slate-800">Coverage Overview</CardTitle>
          <div className="text-xs px-2 py-1 rounded-full bg-blue-50 text-blue-600 border border-blue-100 font-medium">
            {lastUpdated}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-5 pt-2">
        {/* Total Sources with circular visualization */}
        <div className="flex justify-between items-center">
          <div>
            <p className="text-sm font-medium text-slate-500">Total Sources</p>
            <p className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-700">{totalSources}</p>
          </div>
          
          <div className="h-14 w-14 rounded-full bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-100 flex items-center justify-center shadow-sm relative">
            <div className="absolute inset-1 rounded-full border-4 border-t-blue-500 border-r-indigo-500 border-b-purple-500 border-l-transparent animate-spin [animation-duration:3s]"></div>
            <div className="h-8 w-8 rounded-full bg-white flex items-center justify-center font-bold text-sm text-slate-800">
              100%
            </div>
          </div>
        </div>
        
        {/* Divider with gradient */}
        <div className="h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent"></div>
        
        {/* Political Leaning Stats */}
        <div className="space-y-3">
          <p className="text-sm font-medium text-slate-500">Political Leaning Distribution</p>
          
          {/* Left Leaning */}
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-500"></div>
              <span className="text-sm text-slate-700">Left Leaning</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-16 rounded-full bg-slate-100 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-400 to-blue-500" style={{ width: `${leftPercent}%` }}></div>
              </div>
              <span className="px-2 py-0.5 rounded-md text-xs font-medium bg-blue-50 text-blue-600 border border-blue-100">{leaningLeft}</span>
            </div>
          </div>
          
          {/* Center */}
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-purple-500"></div>
              <span className="text-sm text-slate-700">Center</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-16 rounded-full bg-slate-100 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-purple-400 to-purple-500" style={{ width: `${centerPercent}%` }}></div>
              </div>
              <span className="px-2 py-0.5 rounded-md text-xs font-medium bg-purple-50 text-purple-600 border border-purple-100">{center}</span>
            </div>
          </div>
          
          {/* Right Leaning */}
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-indigo-500"></div>
              <span className="text-sm text-slate-700">Right Leaning</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-2 w-16 rounded-full bg-slate-100 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-indigo-400 to-indigo-500" style={{ width: `${rightPercent}%` }}></div>
              </div>
              <span className="px-2 py-0.5 rounded-md text-xs font-medium bg-indigo-50 text-indigo-600 border border-indigo-100">{leaningRight}</span>
            </div>
          </div>
        </div>
        
        {/* Divider with gradient */}
        <div className="h-px bg-gradient-to-r from-transparent via-slate-200 to-transparent"></div>
        
        {/* Bias Distribution Summary */}
        <div className="bg-gradient-to-br from-slate-50 to-white rounded-lg border border-slate-100 p-3">
          <div className="flex justify-between items-center">
            <p className="text-sm font-medium text-slate-700">Bias Distribution</p>
            <div className="px-3 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-sm">
              {biasDistribution}
            </div>
          </div>
          
          {/* Fixed Bias Distribution Chart */}
          <div className="mt-3 h-3 w-full bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full flex">
              <div className="h-full bg-blue-500" style={{ width: `${leftPercent}%` }}></div>
              <div className="h-full bg-purple-500" style={{ width: `${centerPercent}%` }}></div>
              <div className="h-full bg-indigo-500" style={{ width: `${rightPercent}%` }}></div>
            </div>
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-1 px-0.5">
            <span>{leftPercent}%</span>
            <span>{centerPercent}%</span>
            <span>{rightPercent}%</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
