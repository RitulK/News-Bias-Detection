# pip install requests BeautifulSoup newspaper3k spacy
# python -m spacy download en_core_web_md
import requests
import json
from bs4 import BeautifulSoup
import newspaper
from newspaper import Article
import spacy
import re

# Load spaCy model
nlp = spacy.load("en_core_web_md")

sources = {
    "Times of India" : {
        "RSSlink" : "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
        "Country" : "India"
    },

    "NDTV" : {
        "RSSlink" : "https://feeds.feedburner.com/ndtvnews-top-stories",
        "Country" : "India"
    },

    "The Hindu" : {
        "RSSlink" : "https://www.thehindu.com/news/national/feeder/default.rss",
        "Country" : "India"
    },
        
    "The Star Online": {
        "RSSlink": "https://www.thestar.com.my/rss/News/Nation",
        "Country": "Malaysia"
    },
    
    "The Sun (Malaysia)": {
        "RSSlink": "https://thesun.my/rss/local",
        "Country": "Malaysia"
    },
    
    "Free Malaysia Today": {
        "RSSlink": "https://www.freemalaysiatoday.com/category/nation/feed/",
        "Country": "Malaysia"
    },

    "The Guardian Nigeria": {
        "RSSlink": "https://guardian.ng/feed/",
        "Country": "Nigeria"
    },

    "Daily Post (Nigeria)": {
        "RSSlink": "https://dailypost.ng/feed/",
        "Country": "Nigeria"
    },

    "Premium Times": {
        "RSSlink": "https://www.premiumtimesng.com/feed",
        "Country": "Nigeria"
    },

    "The Straits Times": {
        "RSSlink": "https://www.straitstimes.com/news/singapore/rss.xml",
        "Country": "Singapore"
    },

    "Channel NewsAsia": {
        "RSSlink": "https://www.channelnewsasia.com/rssfeeds/8395986",
        "Country": "Singapore"
    },

    "Business Times": {
        "RSSlink": "https://www.businesstimes.com.sg/rss/singapore",
        "Country": "Singapore"
    },

    "CBC News": {
        "RSSlink": "https://www.cbc.ca/webfeed/rss/rss-world",
        "Country": "Canada"
    },

    "Toronto Star": {
        "RSSlink": "https://www.thestar.com/search/?f=rss&t=article&c=news/canada*&l=50&s=start_time&sd=desc",
        "Country": "Canada"
    },

    "Global News (Canada)": {
        "RSSlink": "https://globalnews.ca/feed/",
        "Country": "Canada"
    }
}

# Comprehensive keywords for each news type
news_types = {
    "Political": [
        "election", "vote", "govern", "politic", "parliament", "president", 
        "minist", "democra", "politician", "congress", "senat", "campaign", 
        "part(?:y|ies)", "bill", "legislat", "polic(?:y|ies)", "referendum", "constitut",
        "Democrat", "Republican", "MP", "law(?:s|maker)", "vot(?:e|ing|er)", 
        "ballot", "constituency", "ruling", "opposition", "premier", "chancellery", 
        "diplomat", "prime minister", "ambassador", "constitution", "PMO", "cabinet",
        "assembly", "councilor", "statehouse", "civic", "impeach", "caucus", "SCOTUS"
    ],
    
    "Military & Defense": [
        "military", "army", "navy", "air force", "marine", "defense", "defence", "soldier", 
        "armament", "weapon", "artillery", "squadron", "brigade", "regiment", "battalion", 
        "missile", "nuclear", "strategic", "tactical", "homeland security", "base", "garrison",
        "veteran", "intelligence", "tank", "aircraft carrier", "drone", "radar", "submarine",
        "trooper", "cadet", "admiral", "general", "colonel", "commander", "deployment",
        "NATO", "alliance", "military exercise", "drill", "warfare", "DARPA", "pentagon"
    ],
    
    "Conflict & War": [
        "war", "armed", "attack", "combat", "battle", "bomb", "terror(?:ism|ist)", "conflict", 
        "violence", "invade", "invasion", "guerrilla", "insurgent", "rebel", "militia",
        "civil war", "airstrike", "hostage", "hostility", "genocide", "ceasefire",
        "peace(?:keeping|treaty)", "armistice", "civilian casualt", "refugee", "asylum",
        "humanitarian", "crisis", "bloodshed", "massacre", "confrontation", "occupation",
        "resistance", "uprising", "jihad", "extremist", "warlord", "bunker", "blockade"
    ],
    
    "Crime & Law": [
        "crime", "murder", "homicide", "theft", "robbery", "burglary", "assault", "rape",
        "criminal", "police", "arrest", "law", "legal", "court", "judge", "lawyer", "attorney",
        "prosecution", "defendant", "plaintiff", "jury", "verdict", "sentence", "felony",
        "misdemeanor", "fraud", "corruption", "bribery", "embezzlement", "smuggling", "trafficking",
        "drug", "cartel", "gang", "prison", "jail", "parole", "probation", "investigation",
        "forensic", "warrant", "bail", "convict", "acquit", "trial", "lawsuit", "litigation"
    ],
    
    "Economic & Business": [
        "econom(?:y|ic)", "market", "stock", "trade", "business", "financ", "invest", 
        "inflat", "recession", "gdp", "fiscal", "monetary", "debt", "currency", "budget", 
        "tax", "unemploy", "bank", "commerce", "corporate", "capital", "profit", "revenue",
        "interest rate", "central bank", "fed", "reserve bank", "stock exchange", "dow jones", 
        "nasdaq", "commodity", "export", "import", "tariff", "subsid", "bailout", "austerity",
        "growth", "deficit", "surplus", "wealth", "poverty", "income", "wage", "salary",
        "consumer", "retail", "wholesale", "manufacturing", "industry", "startup", "entrepreneur"
    ],
    
    "Environmental": [
        "climate", "environment", "pollut", "emission", "carbon", "renewable", 
        "sustain", "biodiversity", "ecosystem", "conserv", "wildlife", 
        "forest", "drought", "flood", "hurricane", "earthquake", "disaster", 
        "energy", "green", "recycl", "fossil fuel", "global warming", "greenhouse",
        "solar", "wind power", "hydro(?:electric|power)", "extinction", "endangered",
        "eco(?:logical|system|friendly)", "deforestation", "reforestation", "habitat",
        "organic", "pesticide", "toxic waste", "oil spill", "landfill", "smog",
        "ozone", "acid rain", "desertification", "sustainable development", "biofuel"
    ],
    
    "Health & Medical": [
        "health", "medical", "medicine", "doctor", "hospital", "patient", "disease", 
        "illness", "symptom", "treatment", "therapy", "cure", "vaccine", "vaccination",
        "pandemic", "epidemic", "outbreak", "virus", "bacteria", "infection", "syndrome",
        "cancer", "diabetes", "obesity", "surgery", "pharmaceutical", "drug", "prescription",
        "clinical", "diagnosis", "prognosis", "healthcare", "medicare", "medicaid", "insurance",
        "mental health", "psychology", "psychiatry", "wellness", "nutrition", "diet", 
        "exercise", "public health", "WHO", "CDC", "FDA", "ICU", "ambulance", "emergency"
    ],
    
    "Science & Technology": [
        "science", "technology", "research", "development", "innovation", "experiment",
        "laboratory", "scientist", "engineer", "computer", "software", "hardware", "internet",
        "digital", "data", "algorithm", "AI", "artificial intelligence", "machine learning",
        "robot", "automation", "biotech", "nanotech", "quantum", "physics", "chemistry",
        "biology", "genetics", "DNA", "RNA", "stem cell", "discovery", "breakthrough", 
        "patent", "prototype", "app", "application", "device", "gadget", "tech", "startup",
        "Silicon Valley", "IT", "information technology", "network", "cybersecurity"
    ],
    
    "Sports": [
        "sport", "game", "match", "tournament", "championship", "league", "cup", 
        "football", "soccer", "rugby", "cricket", "tennis", "golf", "basketball", 
        "baseball", "hockey", "boxing", "wrestling", "athlete", "player", "team", 
        "coach", "manager", "stadium", "arena", "Olympic", "medal", "score", "win", 
        "lose", "defeat", "victory", "champion", "record", "fitness", "workout", 
        "race", "racing", "marathon", "swimming", "volleyball", "badminton", "cycling"
    ],
    
    "Celebrity & Entertainment": [
        "celebrity", "star", "famous", "actor", "actress", "singer", "musician", 
        "band", "artist", "entertainer", "Hollywood", "Bollywood", "movie", "film", 
        "cinema", "theater", "TV", "television", "show", "series", "episode", 
        "streaming", "Netflix", "Amazon", "Disney", "HBO", "concert", "performance", 
        "award", "Grammy", "Oscar", "Emmy", "gossip", "scandal", "premiere", "red carpet",
        "celebrity(?:ies)", "superstar", "idol", "fandom", "paparazzi", "tabloid"
    ],
    
    "Social Issues & Human Rights": [
        "social", "society", "community", "human rights", "civil rights", "equality", 
        "inequality", "discrimination", "racism", "sexism", "gender", "LGBTQ", "feminist", 
        "activism", "activist", "protest", "demonstration", "rally", "movement", "reform", 
        "welfare", "poverty", "homeless", "refugee", "immigrant", "migration", "asylum", 
        "labor rights", "worker", "union", "strike", "boycott", "sanction", "censor", 
        "freedom", "liberty", "justice", "injustice", "oppression", "marginalize"
    ],
    
    "Education": [
        "education", "school", "university", "college", "academic", "student", "teacher", 
        "professor", "faculty", "campus", "classroom", "course", "curriculum", "degree", 
        "diploma", "graduate", "undergraduate", "PhD", "thesis", "dissertation", "research", 
        "study", "learning", "teaching", "lecture", "seminar", "workshop", "training", 
        "literacy", "scholarship", "tuition", "enrollment", "admission", "exam", "test", 
        "grade", "GPA", "SAT", "ACT", "GMAT", "GRE", "standardized test", "public education"
    ],
    
    "Infrastructure & Urban Development": [
        "infrastructure", "construction", "building", "architecture", "urban", "city", 
        "town", "municipality", "metropolitan", "housing", "real estate", "property", 
        "development", "redevelopment", "renovation", "restoration", "road", "highway", 
        "bridge", "tunnel", "transit", "transportation", "railway", "airport", "port", 
        "water supply", "sewage", "electricity", "power grid", "telecommunication", 
        "broadband", "internet", "smart city", "public works", "urban planning", "zoning"
    ],
    
    "Space & Astronomy": [
        "space", "astronomy", "cosmos", "universe", "galaxy", "star", "planet", "moon", 
        "solar system", "asteroid", "comet", "meteor", "satellite", "spacecraft", 
        "rocket", "launch", "mission", "astronaut", "cosmonaut", "NASA", "ESA", "SpaceX", 
        "telescope", "observatory", "orbit", "gravity", "black hole", "supernova", 
        "nebula", "pulsar", "quasar", "Mars", "Venus", "Jupiter", "Saturn", "ISS", 
        "International Space Station", "space exploration", "extraterrestrial", "alien"
    ]
}

# Define allowed categories (these categories will be kept, others dropped)
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
    "India": ["India", "Indian", "Indians", "New Delhi", "Delhi", "Mumbai", "Bangalore", "Chennai", 
              "Kolkata", "Hyderabad", "Modi", "BJP", "Congress Party", "Rupee"],
    
    "Canada": ["Canada", "Canadian", "Canadians", "Ottawa", "Toronto", "Vancouver", "Montreal", 
               "Quebec", "Alberta", "Ontario", "Trudeau", "CAD", "Loonie"],
    
    "Singapore": ["Singapore", "Singaporean", "Singaporeans", "SG", "Lion City", 
                 "Changi", "Lee Hsien Loong", "PAP", "SGD"],
    
    "Nigeria": ["Nigeria", "Nigerian", "Nigerians", "Lagos", "Abuja", "Naira", 
                "Buhari", "APC", "PDP"],
    
    "Australia": ["Australia", "Australian", "Australians", "Aussie", "Aussies", "Sydney", 
                  "Melbourne", "Canberra", "Brisbane", "Perth", "AUD", "Albanese"],
    
    "Malaysia": ["Malaysia", "Malaysian", "Malaysians", "Kuala Lumpur", "KL", 
                 "Putrajaya", "Sabah", "Sarawak", "Anwar Ibrahim", "Ringgit", "MYR"],
    
    "United Kingdom": ["United Kingdom", "UK", "Britain", "British", "Britons", "England", 
                      "Scotland", "Wales", "Northern Ireland", "London", "Manchester", 
                      "Birmingham", "Liverpool", "Edinburgh", "Glasgow", "Belfast", 
                      "British", "Pound", "Sterling", "GBP", "Starmer", "Labour", "Tory", "Conservative"]
}

# Flatten the country variations for easier lookup
target_countries = list(country_variations.keys())
country_keywords = {}
for country, variations in country_variations.items():
    for variation in variations:
        country_keywords[variation.lower()] = country

def detect_single_news_type(text):
    """Determine the primary type of news based on keywords and return a single type"""
    text_lower = text.lower()
    
    # Count occurrences of keywords for each type
    type_scores = {news_type: 0 for news_type in news_types.keys()}
    
    # First pass: count keyword matches
    for news_type, keywords in news_types.items():
        for keyword in keywords:
            pattern = r'\b' + keyword + r'(?:\w*)\b'
            matches = re.findall(pattern, text_lower)
            type_scores[news_type] += len(matches)
    
    # If we have a decent text length, enhance with NLP
    if len(text) > 150:
        # Process with spaCy (limit to first 5000 chars for performance)
        doc = nlp(text[:5000])
        
        # Entity type mapping to news categories
        entity_type_map = {
            "PERSON": {"Celebrity & Entertainment": 0.5, "Political": 0.3},
            "ORG": {"Economic & Business": 0.3, "Political": 0.3},
            "GPE": {"Political": 0.3},
            "MONEY": {"Economic & Business": 1.0},
            "PERCENT": {"Economic & Business": 0.8},
            "QUANTITY": {"Economic & Business": 0.5, "Science & Technology": 0.3},
            "LAW": {"Political": 0.5, "Crime & Law": 1.0},
            "NORP": {"Social Issues & Human Rights": 0.5, "Political": 0.3},
            "EVENT": {"Celebrity & Entertainment": 0.3, "Sports": 0.3},
            "PRODUCT": {"Science & Technology": 0.5, "Economic & Business": 0.3},
            "WORK_OF_ART": {"Celebrity & Entertainment": 0.8},
            "FAC": {"Infrastructure & Urban Development": 0.7}
        }
        
        # Enhance scores based on entities
        for ent in doc.ents:
            if ent.label_ in entity_type_map:
                for news_type, weight in entity_type_map[ent.label_].items():
                    type_scores[news_type] += weight
        
        # Check for specific named entities that strongly indicate certain categories
        education_orgs = ["university", "college", "school", "academy", "institute"]
        science_orgs = ["laboratory", "research", "institute", "tech", "science"]
        health_orgs = ["hospital", "clinic", "medical", "healthcare", "health"]
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_name = ent.text.lower()
                if any(edu_term in org_name for edu_term in education_orgs):
                    type_scores["Education"] += 1.5
                if any(sci_term in org_name for sci_term in science_orgs):
                    type_scores["Science & Technology"] += 1.5
                if any(health_term in org_name for health_term in health_orgs):
                    type_scores["Health & Medical"] += 1.5
        
        # Extra check for sports-related content
        sports_indicators = [
            "score", "win", "won", "lost", "defeat", "victory", "champion", "tournament",
            "match", "game", "league", "team", "player", "coach"
        ]
        sports_count = sum(1 for term in sports_indicators if re.search(r'\b' + term + r'\b', text_lower))
        if sports_count >= 3:
            type_scores["Sports"] += sports_count * 0.5
    
    # Pick the highest scoring type
    if any(type_scores.values()):
        primary_type = max(type_scores.items(), key=lambda x: x[1])[0]
        # Only return if we have meaningful evidence
        if type_scores[primary_type] > 0:
            return primary_type
    
    # Fall back to a generic category based on length and structure
    # If no clear category detected, make educated guess based on text characteristics
    if len(text) > 500:  # For longer texts
        sentences = text.split('.')
        if any(re.search(r'\b\d+%|\$\d+|\d+ percent\b', text_lower)):
            return "Economic & Business"
        elif any(re.search(r'\bcourt\b|\bjudge\b|\bpolice\b|\barrest', text_lower)):
            return "Crime & Law"
        elif len(sentences) > 10 and any(len(s.split()) > 20 for s in sentences):
            return "Political"  # Political articles tend to be verbose
        else:
            return "Social Issues & Human Rights"  # Default for substantial content
    else:
        return "Social Issues & Human Rights"  # Default fallback category

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

def articleScraper(sourceURL, source, country):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(sourceURL, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, "xml")
        i = 0
        
        for item in soup.find_all("item"):
            try:
                # Get the link text safely
                link_text = item.link.text if item.link else ""
                if not link_text:
                    continue
                    
                article = Article(link_text)
                article.download()
                article.parse()
                
                # Get article content and title
                content = article.text
                title = item.title.text if item.title else "No title"
                
                # First check title alone for efficiency
                title_news_type = detect_single_news_type(title)
                title_countries = detect_countries(title)
                
                # If we already have matches from title, don't process the full content
                if title_news_type and (title_countries or country in target_countries):
                    news_type_detected = title_news_type
                    countries_mentioned = title_countries
                else:
                    # Process full content
                    combined_text = title + " " + content
                    news_type_detected = detect_single_news_type(combined_text)
                    countries_mentioned = detect_countries(combined_text)
                
                # Always include the source country in countries mentioned
                if country in target_countries and country not in countries_mentioned:
                    countries_mentioned.append(country)
                
                # Check if news type is in allowed categories
                if news_type_detected and news_type_detected in allowed_categories and countries_mentioned:
                    items.append({
                        "title": title,
                        "link": link_text,
                        "description": item.description.text if hasattr(item, 'description') and item.description else "No description",
                        "published_date": item.pubDate.text if hasattr(item, 'pubDate') and item.pubDate else "No date",
                        "source": source,
                        "source_country": country,
                        "news_type": news_type_detected,
                        "countries_mentioned": countries_mentioned,
                        "content": content[:1000] + ("..." if len(content) > 1000 else "")  # Truncate content
                    })
                    i = i + 1
                    
                # Limiting to 5 Articles per Publication
                if i == 5:
                    break
                    
            except Exception as e:
                print(f"Error processing article from {source}: {str(e)}")
                continue
                
        print(f"Scraped {i} Articles from {source}")
    except Exception as e:
        print(f"Error accessing RSS feed for {source}: {str(e)}")

items = []

for source in sources:
    try:
        articleScraper(sources[source]['RSSlink'], source, sources[source]['Country'])
    except Exception as e:
        print(f"Error scraping from {source}: {str(e)}")

# Articles are already filtered during scraping
filtered_items = items

articles = json.dumps(filtered_items, indent=4)
print(articles)

# # Optionally save to file
with open("filtered_news_articles.json", "w") as f:
    f.write(articles)
print("\nFiltered news articles have been saved to 'filtered_news_articles.json'.")

#get filteration report 
print(f"\nTotal articles filtered: {len(filtered_items)}")
print("News types distribution:")
type_counts = {}
for item in filtered_items:
    news_type = item["news_type"]
    type_counts[news_type] = type_counts.get(news_type, 0) + 1
for news_type, count in sorted(type_counts.items()):
    print(f"- {news_type}: {count}")

# print("\nCountries mentioned distribution:")
# country_counts = {}
# for item in filtered_items:
#     for country in item["countries_mentioned"]:
#         country_counts[country] = country_counts.get(country, 0) + 1
# for country, count in sorted(country_counts.items()):
#     print(f"- {country}: {count}")

# # Display filtered out categories
# print("\nFiltered categories (articles will be dropped from these):")
# excluded_categories = [cat for cat in news_types.keys() if cat not in allowed_categories]
# for category in sorted(excluded_categories):
#     print(f"- {category}")