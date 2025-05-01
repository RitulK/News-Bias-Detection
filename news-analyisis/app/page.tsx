import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"

interface NewsHeadline {
  _id: string;
  headline: string;
}

async function getHeadlines() {
  try {
    const response = await fetch("http://localhost:8000/eventheadlines", {
      cache: "no-store" // Ensures fresh data on each request
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch headlines: ${response.status}`);
    }
    
    const data = await response.json();
    // Take only first 5 headlines
    return data.slice(0, 5);
  } catch (error) {
    console.error("Failed to fetch headlines:", error);
    return [];
  }
}

function HeadlineCard({ headline }: { headline: NewsHeadline }) {
  return (
    <Card className="overflow-hidden h-full flex flex-col transition-all duration-300 hover:shadow-xl hover:scale-[1.02] border border-slate-200 relative backdrop-blur-sm bg-white/90">
      <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-br from-blue-500/10 to-indigo-500/20 rounded-bl-full -z-10"></div>
      <div className="absolute -bottom-4 -left-4 w-12 h-12 rounded-full bg-blue-500/5 -z-10"></div>
      
      <CardHeader className="pb-2 border-b border-slate-100">
        <div className="w-8 h-1 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full mb-3"></div>
        <CardTitle className="line-clamp-2 text-lg font-medium text-slate-800">{headline.eventHeadline}</CardTitle>
      </CardHeader>
      
      <CardContent className="flex-grow pt-4">
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <span className="inline-block w-2 h-2 bg-indigo-500 rounded-full opacity-75"></span>
          ID: {headline._id.substring(0, 8)}...
        </p>
      </CardContent>
      
      <CardFooter className="pt-2 pb-4">
        <Button asChild className="rounded-full transition-all duration-200 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md hover:shadow-lg hover:shadow-blue-500/20 w-full">
          <Link href={`/event/${headline._id}`}>
            <span className="flex items-center justify-center gap-2">
              View Details
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="ml-1">
                <path d="M5 12h14"></path>
                <path d="m12 5 7 7-7 7"></path>
              </svg>
            </span>
          </Link>
        </Button>
      </CardFooter>
    </Card>
  )
}

export default async function Home() {
  const headlines = await getHeadlines();
  const hasHeadlines = headlines.length > 0;

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Enhanced Background with darker colors */}
      <div className="fixed inset-0 bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 -z-20"></div>
      
      {/* Subtle grid pattern overlay with higher opacity */}
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGZpbGw9InRyYW5zcGFyZW50IiBkPSJNMCAwaDYwdjYwSDB6Ii8+PHBhdGggZD0iTTM2IDM0aC0ydi00aDJ2NHptMC04aC0ydi00aDJ2NHpNMjQgMzBoLTR2Mmg0di0yem04LTRoLTR2Mmg0di0yeiIgZmlsbD0iI2JiYzVlMSIgZmlsbC1ydWxlPSJub256ZXJvIi8+PC9nPjwvc3ZnPg==')] opacity-50 -z-10"></div>
      
      {/* Animated blobs with higher opacity and richer colors */}
      <div className="fixed top-20 right-20 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob -z-10"></div>
      <div className="fixed bottom-20 left-20 w-96 h-96 bg-indigo-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-2000 -z-10"></div>
      <div className="fixed top-1/3 left-1/3 w-96 h-96 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-blob animation-delay-4000 -z-10"></div>
      
      {/* Gradient beams with higher opacity */}
      <div className="fixed -top-24 -right-24 w-96 h-96 bg-gradient-to-br from-blue-500 to-indigo-600 opacity-20 rotate-12 transform -z-10"></div>
      <div className="fixed -bottom-24 -left-24 w-96 h-96 bg-gradient-to-tr from-indigo-500 to-purple-600 opacity-20 -rotate-12 transform -z-10"></div>
      
      {/* Additional color elements */}
      <div className="fixed bottom-1/3 right-1/4 w-64 h-64 bg-teal-300 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-3000 -z-10"></div>
      <div className="fixed top-2/3 left-1/4 w-48 h-48 bg-pink-200 rounded-full mix-blend-multiply filter blur-3xl opacity-15 animate-blob animation-delay-5000 -z-10"></div>
      
      {/* Subtle texture overlay */}
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI1IiBoZWlnaHQ9IjUiPgo8cmVjdCB3aWR0aD0iNSIgaGVpZ2h0PSI1IiBmaWxsPSIjZmZmZmZmMTAiPjwvcmVjdD4KPHBhdGggZD0iTTAgNUw1IDBaTTYgNEw0IDZaTS0xIDFMMSAtMVoiIHN0cm9rZT0iIzg4ODg4ODEwIiBzdHJva2Utd2lkdGg9IjEiPjwvcGF0aD4KPC9zdmc+')] opacity-25 -z-10"></div>
      
      {/* Content containers with adjusted backgrounds */}
      <div className="container mx-auto py-20 px-6 relative z-10">
        {/* Hero section with refined design */}
        <div className="mb-12 text-center mt-4 backdrop-blur-md py-12 rounded-3xl bg-gradient-to-b from-white/80 to-white/60 border border-white/50 shadow-2xl shadow-blue-900/10">
          <div className="inline-block mb-5 px-5 py-2.5 rounded-full bg-blue-50/80 text-blue-700 text-sm font-medium border border-blue-100/80 shadow-sm backdrop-blur-sm">
            <span className="flex items-center gap-2.5">
              <span className="inline-block w-2.5 h-2.5 bg-blue-500 rounded-full animate-pulse"></span>
              News Analysis Tool
            </span>
          </div>
          <h1 className="text-5xl md:text-6xl font-black bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-700 tracking-tight mb-4 px-4">News Bias Detection</h1>
          <div className="h-1.5 w-24 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full mx-auto my-5"></div>
          <p className="text-slate-700 mt-5 text-xl max-w-2xl mx-auto font-medium">Explore top news headlines and analyze their potential bias with AI-powered insights</p>
        </div>

        {/* Main content with more sophisticated design */}
        <main className="mx-auto relative backdrop-blur-md py-12 px-10 rounded-3xl bg-gradient-to-br from-white/70 to-white/60 border border-white/50 shadow-2xl shadow-blue-900/10">
          <div className="flex justify-between items-center mb-14">
            <div className="flex flex-col">
              <div className="flex items-center gap-3 mb-2">
                <span className="inline-block w-3 h-10 bg-gradient-to-b from-blue-600 to-indigo-600 rounded-full"></span>
                <h2 className="text-4xl font-bold text-slate-800">Latest Headlines</h2>
              </div>
              <p className="text-slate-500 ml-6">Discover trending news from multiple sources</p>
            </div>
            <div className="flex gap-2">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-1.5 w-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-full"></div>
              ))}
            </div>
          </div>
          
          {!hasHeadlines ? (
            <div className="bg-white/80 text-destructive p-10 rounded-2xl mb-6 shadow-lg border border-destructive/10 text-center backdrop-blur-md">
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-destructive/10 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-destructive">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
              </div>
              <h3 className="text-xl font-semibold mb-2">Unable to Load Headlines</h3>
              <p>Failed to load headlines. Please try again later.</p>
            </div>
          ) : (
            <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3">
              {headlines.map((headline: any) => (
                <HeadlineCard key={headline._id} headline={headline} />
              ))}
            </div>
          )}
        </main>
        
        {/* Optional brand footer */}
        <div className="mt-12 text-center">
          <div className="inline-flex items-center gap-2 py-2.5 px-5 rounded-full bg-white/40 backdrop-blur-sm border border-white/30 shadow-sm">
            <div className="h-3 w-3 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600"></div>
            <p className="text-sm text-slate-600 font-medium">Powered by AI · Reliable Analysis · Updated Daily</p>
          </div>
        </div>
      </div>
    </div>
  )
}