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
    <Card className="overflow-hidden h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="line-clamp-2">{headline.eventHeadline}</CardTitle>
      </CardHeader>
      <CardContent className="flex-grow">
        <p className="text-sm text-muted-foreground">
          ID: {headline._id.substring(0, 8)}...
        </p>
      </CardContent>
      <CardFooter>
        <Button asChild>
          <Link href={`/event/${headline._id}`}>
            View Details
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
    <div className="container mx-auto py-8">
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold">News Bias Detection</h1>
        <p className="text-muted-foreground mt-2">Explore top news headlines and analyze their bias</p>
      </header>

      <main>
        <h2 className="text-2xl font-semibold mb-4">Latest Headlines</h2>
        
        {!hasHeadlines ? (
          <div className="bg-destructive/15 text-destructive p-4 rounded-md mb-4">
            Failed to load headlines. Please try again later.
          </div>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {headlines.map((headline: any) => (
              <HeadlineCard key={headline._id} headline={headline} />
            ))}
          </div>
        )}
      </main>
    </div>
  )
}