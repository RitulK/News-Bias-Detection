import Summary from "./Summary";
import NewsArticles from "./ArticleList";
import CoverageDetails from "./CoverageDetails";



export default async function Home({ params }: { params: { eventID: string } }) {
      const resolvedParams = await params;
      const eventID = resolvedParams.eventID;
      console.log("Event ID:", eventID); // Log the event ID to verify it's being passed correctly
      const eventDataResponse = await fetch(`http://localhost:8000/event/${eventID}`);
      const eventData = await eventDataResponse.json();
      console.log("Event Data:", eventData); // Log the event data to verify it's being fetched correctly
      // Now you can access eventData properties
      const event = eventData;
      // const event = {
      //   "publishedDate": "1 day ago",
      //   "location": "Singapore",
      //   "updateTime": "4 hours ago",
      //   "eventHeadline": "Singapore detains teenage boy allegedly planning to kill Muslims",
      //   "leftSummary": "A 17-year-old Singaporean boy was detained for allegedly planning to kill dozens of Muslims outside mosques, according to the Internal Security Department. The boy idolized Brenton Tarrant, a perpetrator of the 2019 New Zealand mosque attacks, and wanted to kill at least 100 Muslims. Home Minister K Shanmugam expressed concern about the rise of radicalism among youth via the internet, noting that self-radicalization can occur rapidly. Authorities reported that a 15-year-old girl had been restricted for online relationships with ISIS supporters.",
      //   "centerSummary": "This is a neutral perspective summarizing the key details of the incident without emphasizing a particular bias.",
      //   "rightSummary": "This section presents an alternative viewpoint or additional context that might not be covered in other perspectives."
      // }
      
      const articles = [
        {
          id: 1,
          source: "Jerusalem Post",
          sourceLogo: "/logos/jerusalem_post.png",
          title: "Singapore detains two teens, mosque shootings, ISIS ties",
          summary: "Singapore detained two teenagers under internal security laws for extremist activities...",
          timestamp: "4 hours ago",
          location: "Jerusalem, Israel",
          alignment: "Left", // ✅ Now correctly typed
          link: "https://example.com/article1",
        },
        {
          id: 2,
          source: "The Express Tribune",
          sourceLogo: "/logos/express_tribune.png",
          title: "Singapore arrests two teenagers under security law for extremist plans",
          summary: "Singapore has detained two teenagers under its Internal Security Act (ISA)...",
          timestamp: "22 hours ago",
          location: "Pakistan",
          alignment: "Center", // ✅ Now correctly typed
          link: "https://example.com/article2",
        },
        {
          id: 3,
          source: "phelpscountyfocus.com",
          sourceLogo: "/logos/phelpscounty.png",
          title: "Singapore detains teenage boy allegedly planning to kill Muslims",
          summary: "A Singaporean teenager allegedly planning to kill dozens of Muslims...",
          timestamp: "1 day ago",
          location: "Singapore",
          alignment: "Right", // ✅ Now correctly typed
          link: "https://example.com/article3",
        }
      ];


      
      
    return (
    <div className="w-full min-h-screen p-6 flex gap-6 mt-12">
    {/* Left Column */}
    <div className="w-2/3">
    <Summary event={event}/>
    <NewsArticles articles={articles}/>
    </div>

    {/* Right Column */}
    <div className="w-1/3 mt-12">
    <h2 className="text-xl font-bold">News Analysis</h2>
    <CoverageDetails
  totalSources={44}
  leaningLeft={3}
  leaningRight={2}
  center={8}
  lastUpdated="4 hours ago"
  biasDistribution="62% Center"
/>
    </div>
  </div>
  );
}
