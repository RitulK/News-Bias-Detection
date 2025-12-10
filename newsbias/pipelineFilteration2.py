# pip install requests BeautifulSoup newspaper3k spacy google-genai
# python -m spacy download en_core_web_md
import requests
import json
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import spacy
import re
import google.generativeai as genai
import os
from time import sleep
from datetime import datetime, timedelta
from collections import defaultdict

# Initialize Gemini client
GOOGLE_API_KEY = "GOOGLE_API_KEY"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.0-flash"  # Using the latest model for better accuracy

# Configuration for the optimized pipeline
GEMINI_RATE_LIMIT = 15  # Max requests per minute

# Load spaCy model
nlp = spacy.load("en_core_web_md")

sources = {
    "Times of India": {
        "RSSlink": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "Country": "India"
    },
    "NDTV": {
        "RSSlink": "https://feeds.feedburner.com/ndtvnews-top-stories",
        "Country": "India"
    },
    "The Hindu": {
        "RSSlink": "https://www.thehindu.com/news/national/feeder/default.rss",
        "Country": "India"
    },
    
    "Mint Politics" : {
        "RSSlink" : "https://www.livemint.com/rss/politics",
        "Country" : "India"
    },

    "Mint News" : {
        "RSSlink" : "https://www.livemint.com/rss/news",
        "Country" : "India"
    },

    "CNBC Politics" : {
        "RSSlink" : "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/politics.xml",
        "Country" : "India"
    },

    "CNBC Economics" : {
        "RSSlink" : "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/economy.xml",
        "Country" : "India"
    },

    "CNBC World" : {
        "RSSlink" : "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/world.xml",
        "Country" : "India"
    },

    "CNBC Market" : {
        "RSSlink" : "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
        "Country" : "India"
    },

    "DNA India" : {
        "RSSlink" : "https://www.dnaindia.com/feeds/india.xml",
        "Country" : "India"
    },
    "CNN": {
        "RSSlink": "http://rss.cnn.com/rss/cnn_world.rss",
        "Country": "USA"
    },
    
    "NewYorkTimes": {
        "RSSlink": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "Country": "USA"
    },

    "The Sydney Morning Herald": {
        "RSSlink": "https://www.smh.com.au/rss/feed.xml",
        "Country": "Australia"
    },

    "Independent Australia": {
        "RSSlink": "http://feeds.feedburner.com/IndependentAustralia",
        "Country": "Australia"
    },

    "The Age": {
        "RSSlink": "https://www.theage.com.au/rss/feed.xml",
        "Country": "Australia"
    },
}

# Define allowed categories
allowed_categories = [
    "Political",
    "Military & Defense",
    "Conflict & War",
    "Crime & Law",
    "Economic & Business",
    "Environmental",
    "Social Issues & Human Rights"
]

# List of countries to filter for - add variations, abbreviations, adjectives, demonyms
country_variations = {
    "India": ["India", "Indian", "Indians", "New Delhi", "Delhi", "Mumbai", "Bangalore", "Bengaluru", "Chennai",
"Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Modi", "Narendra Modi", 
"Rahul Gandhi", "Sonia Gandhi", "BJP", "Congress Party", "Aam Aadmi Party", "AAP", "RSS", 
"Rupee", "INR", "Gandhi", "Lok Sabha", "Rajya Sabha", "Parliament", "Holi", "Diwali", "Bollywood", "Indian Army", "ISRO", "Tata", "Reliance", "Adani", "Reserve Bank of India", "RBI", 
"Make in India", "Swachh Bharat", "Uttar Pradesh", "Maharashtra", "Kerala", "Tamil Nadu"],
    
    "Australia": ["Australia", "Australian", "Australians", "Aussie", "Aussies", "Sydney", "Melbourne", "Canberra", 
"Brisbane", "Perth", "Adelaide", "Hobart", "Darwin", "Albanese", "Anthony Albanese", 
"Scott Morrison", "Labor Party", "Liberal Party", "Australian Dollar", "AUD", "Down Under", 
"Outback", "Kangaroo", "Cricket Australia", "ANZAC", "Australian Open", "Great Barrier Reef", 
"ABC News Australia", "Commonwealth", "NSW", "Victoria", "Queensland", "Western Australia"],
    
    "USA": ["USA", "United States", "America", "Americans", "US", "Washington", "Washington D.C.", 
"New York", "Los Angeles", "Chicago", "Houston", "San Francisco", "Seattle", "Boston", 
"Trump", "Donald Trump", "Biden", "Joe Biden", "Democrats", "Republicans", "White House", 
"Capitol Hill", "Congress", "Senate", "USD", "Dollar", "Federal Reserve", "Wall Street", 
"Silicon Valley", "Hollywood", "NYPD", "CIA", "FBI", "Thanksgiving", "Black Friday", 
"Fourth of July", "NASA", "Apple", "Google", "Amazon", "Microsoft", "Super Bowl"]
}

# Flatten the country variations for easier lookup
target_countries = list(country_variations.keys())
country_keywords = {}
for country, variations in country_variations.items():
    for variation in variations:
        country_keywords[variation.lower()] = country

class RateLimiter:
    def __init__(self):
        self.calls = []
    
    def wait(self):
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if (now - call).total_seconds() < 60]
        
        if len(self.calls) >= GEMINI_RATE_LIMIT:
            oldest = self.calls[0]
            wait_time = (60 - (now - oldest).total_seconds()) + 0.1
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
            sleep(wait_time)
        
        self.calls.append(datetime.now())

limiter = RateLimiter()

def detect_countries(text):
    """Detect mentions of target countries in the text"""
    text_lower = text.lower()
    mentioned_countries = set()
    
    # First try direct keyword matching (efficient)
    for keyword, country in country_keywords.items():
        # Use word boundary matching to avoid partial matches
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            mentioned_countries.add(country)
    
    # If we have a decent amount of text, also use spaCy for geopolitical entities
    if len(text) > 100 and len(mentioned_countries) == 0:
        # Process with spaCy, limiting text length for performance
        doc = nlp(text[:5000])
        
        # Extract locations from named entities
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                # Check if this entity matches any of our target countries or their variations
                ent_text = ent.text.lower()
                for keyword, country in country_keywords.items():
                    # Allow partial matches for countries and major cities
                    if keyword in ent_text or ent_text in keyword:
                        mentioned_countries.add(country)
    
    return list(mentioned_countries)

def gemini_classify_article(article):
    """Classify article content using Gemini"""
    try:
        limiter.wait()
        
        # Create prompt text
        prompt = f"""Classify this news article into exactly one of these categories: 
        {', '.join(allowed_categories)}. 
        Return ONLY the category name, nothing else.
        
        Title: {article['title']}
        Content: {article['content'][:2000]}"""
        
        # Correct API call format
        model = genai.GenerativeModel(MODEL_ID)
        response = model.generate_content(prompt)
        
        # Parse response safely
        if hasattr(response, 'text'):
            category = response.text.strip()
            if category in allowed_categories:
                return category
        return None
        
    except Exception as e:
        print(f"⚠️ Gemini error (will retry): {str(e)}")
        sleep(2)
        return gemini_classify_article(article)  # Retry

def articleScraper(sourceURL, source, country):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(sourceURL, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, "xml")
        scraped_articles = []
        
        for item in soup.find_all("item"):
            try:
                link_text = item.link.text if item.link else ""
                if not link_text:
                    continue
                    
                article = Article(link_text)
                article.download()
                article.parse()
                
                content = article.text
                title = item.title.text if item.title else "No title"
                
                # Check if article mentions target countries
                countries_mentioned = detect_countries(title + " " + content)
                if not countries_mentioned and country not in target_countries:
                    continue
                
                # Always include the source country
                if country in target_countries and country not in countries_mentioned:
                    countries_mentioned.append(country)
                
                scraped_articles.append({
                    "title": title,
                    "link": link_text,
                    "description": item.description.text if hasattr(item, 'description') and item.description else "No description",
                    "published_date": item.pubDate.text if hasattr(item, 'pubDate') and item.pubDate else "No date",
                    "source": source,
                    "source_country": country,
                    "countries_mentioned": countries_mentioned,
                    "content": content
                })
                
                if len(scraped_articles) >= 5:  # Limit to 5 articles per source
                    break
                    
            except Exception as e:
                print(f"Error processing article from {source}: {str(e)}")
                continue
                
        print(f"Scraped {len(scraped_articles)} Articles from {source}")
        return scraped_articles
        
    except Exception as e:
        print(f"Error accessing RSS feed for {source}: {str(e)}")
        return []

def process_articles(articles):
    """Process articles with Gemini classification"""
    print(f"\n🔍 Starting processing of {len(articles)} articles...")
    
    results = {
        'country_rejected': 0,
        'sent_to_gemini': 0,
        'gemini_approved': 0,
        'final_counts': defaultdict(int)
    }
    
    filtered_articles = []
    
    for idx, article in enumerate(articles, 1):
        # Skip if no target countries mentioned
        if not article['countries_mentioned']:
            results['country_rejected'] += 1
            continue
            
        results['sent_to_gemini'] += 1
        category = gemini_classify_article(article)
        
        if category:
            article['news_type'] = category
            filtered_articles.append(article)
            results['final_counts'][category] += 1
            results['gemini_approved'] += 1
            print(f"✅ {idx}/{len(articles)}: {category} - {article['title'][:50]}...")
        else:
            print(f"❌ {idx}/{len(articles)}: Rejected by Gemini")
    
    # Reporting
    print("\n📊 Final Results:")
    print(f"Articles processed: {len(articles)}")
    print(f"Rejected by country filter: {results['country_rejected']}")
    print(f"Sent to Gemini: {results['sent_to_gemini']}")
    print(f"Approved by Gemini: {results['gemini_approved']}")
    print("\nApproved articles by category:")
    for cat, count in results['final_counts'].items():
        print(f"- {cat}: {count}")
    
    return filtered_articles

def main():
    all_articles = []
    
    # Scrape articles from all sources
    for source in sources:
        try:
            articles = articleScraper(sources[source]['RSSlink'], source, sources[source]['Country'])
            all_articles.extend(articles)
        except Exception as e:
            print(f"Error scraping from {source}: {str(e)}")
    
    if not all_articles:
        print("No articles were scraped.")
        return
    
    # Process articles with Gemini classification
    print("\nProcessing articles with Gemini...")
    filtered_articles = process_articles(all_articles)
    
    # Save results
    with open("filtered_news_articles1.json", "w") as f:
        json.dump(filtered_articles, f, indent=4)
    
    print("\nFiltered news articles have been saved to 'filtered_news_articles1.json'")

if __name__ == "__main__":
    main()
