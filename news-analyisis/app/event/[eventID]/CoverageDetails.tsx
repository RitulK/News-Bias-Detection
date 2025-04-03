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

  return (
    <Card className="bg-white border border-black rounded-lg p-5 mt-4">
      <CardHeader>
        <CardTitle className="text-xl font-semibold text-black">Coverage Overview</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-gray-800">
        <div className="flex justify-between items-center text-lg font-medium">
          <span>Total News Sources</span>
          <span className="text-black">{totalSources}</span>
        </div>
        <div className="flex justify-between items-center text-base">
          <span>Leaning Left</span>
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-red-600 text-white">{leaningLeft}</span>
        </div>
        <div className="flex justify-between items-center text-base">
          <span>Leaning Right</span>
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-purple-600 text-white">{leaningRight}</span>
        </div>
        <div className="flex justify-between items-center text-base">
          <span>Center</span>
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-green-600 text-white">{center}</span>
        </div>
        <hr className="border-black my-2" />
        <div className="flex justify-between items-center text-base mt-8">
          <span>Last Updated</span>
          <span className="text-black font-medium">{lastUpdated}</span>
        </div>
        <div className="flex justify-between items-center text-base">
          <span>Bias Distribution</span>
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-orange-600 text-white">{biasDistribution}</span>
        </div>
        <hr className="border-black my-2" />

      </CardContent>
    </Card>
  );
}
