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
      <div className="min-h-screen relative overflow-hidden">
        {/* Background elements */}
        <div className="fixed inset-0 bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 -z-20"></div>
        <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGZpbGw9InRyYW5zcGFyZW50IiBkPSJNMCAwaDYwdjYwSDB6Ii8+PHBhdGggZD0iTTM2IDM0aC0ydi00aDJ2NHptMC04aC0ydi00aDJ2NHpNMjQgMzBoLTR2Mmg0di0yem04LTRoLTR2Mmg0di0yeiIgZmlsbD0iI2JiYzVlMSIgZmlsbC1ydWxlPSJub256ZXJvIi8+PC9nPjwvc3ZnPg==')] opacity-50 -z-10"></div>
        
        {/* Animated blobs */}
        <div className="fixed top-20 right-20 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob -z-10"></div>
        <div className="fixed bottom-20 left-20 w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000 -z-10"></div>
        
        {/* Main content */}
        <div className="container mx-auto pt-24 pb-16 px-6 relative z-10">
          <div className="flex flex-col md:flex-row gap-8">
            {/* Left Column */}
            <div className="w-full md:w-2/3">
              <div className="backdrop-blur-md rounded-3xl bg-white/60 border border-white/50 shadow-xl shadow-blue-900/5 overflow-hidden mb-8">
                <Summary event={event} />
              </div>
              
              <div className="backdrop-blur-md rounded-3xl bg-white/60 border border-white/50 shadow-xl shadow-blue-900/5 p-8">
                <div className="mb-6 flex justify-between items-center">
                  <div className="flex items-center gap-3">
                    <span className="inline-block w-2 h-8 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full"></span>
                    <h2 className="text-2xl font-bold text-slate-800">Related Articles</h2>
                  </div>
                  <div className="flex gap-1">
                    {[1, 2, 3].map(i => (
                      <div key={i} className="h-1 w-5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full"></div>
                    ))}
                  </div>
                </div>
                <NewsArticles articles={articles} />
              </div>
            </div>

            {/* Right Column */}
            <div className="w-full md:w-1/3 space-y-8">
              <div className="backdrop-blur-md rounded-3xl bg-white/60 border border-white/50 shadow-xl shadow-blue-900/5 p-8">
                <div className="mb-6 flex items-center gap-3">
                  <span className="inline-block w-2 h-6 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full"></span>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-indigo-700">News Analysis</h2>
                </div>
                <CoverageDetails
                  totalSources={44}
                  leaningLeft={3}
                  leaningRight={2}
                  center={8}
                  lastUpdated="4 hours ago"
                  biasDistribution="62% Center"
                />
              </div>
              
              {/* Additional info card */}
              <div className="backdrop-blur-md rounded-3xl bg-gradient-to-br from-blue-600/10 to-indigo-600/10 border border-white/30 shadow-xl shadow-blue-900/5 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
                      <circle cx="12" cy="12" r="10"></circle>
                      <line x1="12" y1="8" x2="12" y2="12"></line>
                      <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                  </div>
                  <h3 className="font-medium text-slate-800">About This Analysis</h3>
                </div>
                <p className="text-sm text-slate-600">
                  Our AI analyzes multiple news sources to identify potential bias in coverage. 
                  Results are categorized by political leaning to provide a balanced perspective.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
}
