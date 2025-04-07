import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
// import { NewsHeadline } from "@/types/news"

interface NewsHeadline {
    _id: string;
    headline: string;
}

interface HeadlineCardProps {
  headline: NewsHeadline
}

export default function HeadlineCard({ headline }: HeadlineCardProps) {
  return (
    <Card className="overflow-hidden h-full flex flex-col">
      <CardHeader className="pb-2">
        <CardTitle className="line-clamp-2">{headline.headline}</CardTitle>
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