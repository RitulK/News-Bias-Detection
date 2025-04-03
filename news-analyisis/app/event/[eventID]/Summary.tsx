"use client"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

interface Event {
  publishedDate: string;
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
  const { publishedDate, location, updateTime, eventHeadline, leftSummary, centerSummary, rightSummary } = event;

  return (
    <div className="w-full mx-auto p-4">
      <p className="text-sm text-gray-500">
        Published {publishedDate} • <span className="text-blue-500 cursor-pointer">{location}</span> • Updated {updateTime}
      </p>
      <Card className="mt-2 border-black">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-black">{eventHeadline}</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="left">
            <TabsList className="flex justify-start border border-black rounded-lg bg-white p-1">
              <TabsTrigger
                value="left"
                className="data-[state=active]:bg-black data-[state=active]:text-white px-4 py-2 rounded-md transition-colors"
              >
                Left
              </TabsTrigger>
              <TabsTrigger
                value="center"
                className="data-[state=active]:bg-black data-[state=active]:text-white px-4 py-2 rounded-md transition-colors"
              >
                Center
              </TabsTrigger>
              <TabsTrigger
                value="right"
                className="data-[state=active]:bg-black data-[state=active]:text-white px-4 py-2 rounded-md transition-colors"
              >
                Right
              </TabsTrigger>
            </TabsList>

            <TabsContent value="left">
              <p className="text-gray-700 mt-4">{leftSummary}</p>
            </TabsContent>
            <TabsContent value="center">
              <p className="text-gray-700 mt-4">{centerSummary}</p>
            </TabsContent>
            <TabsContent value="right">
              <p className="text-gray-700 mt-4">{rightSummary}</p>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
