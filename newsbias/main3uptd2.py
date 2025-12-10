from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime, timezone
import google.generativeai as genai
from time import sleep
from collections import defaultdict
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import hashlib
import os
from bson import json_util

# Configure Gemini API
GOOGLE_API_KEY = "Google_API_Key"
genai.configure(api_key=GOOGLE_API_KEY)
EVENT_MODEL = "gemini-2.0-flash"

# MongoDB Configuration
MONGO_URI = "MONGO_URI"
DATABASE_NAME = "news_analysis"
EVENTS_COLLECTION = "events"
ARTICLES_COLLECTION = "articles"
SOURCES_COLLECTION = "news_source1"

# RateLimiter class for Gemini API
class RateLimiter:
    def __init__(self, max_calls=12):
        self.calls = []
        self.max_calls = max_calls
    
    def wait(self):
        now = datetime.now(timezone.utc)
        self.calls = [call for call in self.calls if (now - call).total_seconds() < 60]
        
        if len(self.calls) >= self.max_calls:
            oldest = self.calls[0]
            wait_time = (60 - (now - oldest).total_seconds()) + 0.1
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
            sleep(wait_time)
        
        self.calls.append(now)

limiter = RateLimiter()

# Initialize models
SIMILARITY_THRESHOLD = 0.65
similarity_model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")
bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)
print("All models loaded successfully")

class EventClusterer:
    def __init__(self):
        self.similarity_model = similarity_model
        self.bias_pipeline = bias_pipeline
        self.events = []
        self.gemini_cache = {}
        
        # Initialize MongoDB connection
        self.mongo_client = None
        self.db = None
        self.events_col = None
        self.articles_col = None
        self.sources_col = None
        self._init_mongo()
        
        self._load_events_from_mongo()
    
    def _init_mongo(self):
        """Initialize MongoDB connection and collections"""
        try:
            self.mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
            self.mongo_client.admin.command('ping')
            print("Successfully connected to MongoDB!")
            
            self.db = self.mongo_client[DATABASE_NAME]
            self.events_col = self.db[EVENTS_COLLECTION]
            self.articles_col = self.db[ARTICLES_COLLECTION]
            self.sources_col = self.db[SOURCES_COLLECTION]
            
            # Create indexes
            self.events_col.create_index("event_id", unique=True)
            self.articles_col.create_index("link", unique=True)
            self.articles_col.create_index("event_id")
            self.sources_col.create_index("source_name", unique=True)
            self.sources_col.create_index("source_id", unique=True)
            
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise
    
    def _update_source_stats(self, article):
        """Update or create source statistics in MongoDB"""
        source_name = article.get("source", "Unknown")
        source_country = article.get("source_country", "Unknown")
        alignment = article.get("alignment", "center")
        
        # Map lowercase alignment to title case for MongoDB
        alignment_map = {
            "left": "Left",
            "center": "Center", 
            "right": "Right"
        }
        
        # Use title case version of alignment for MongoDB
        mongodb_alignment = alignment_map.get(alignment, "Center")
        
        # Generate consistent source_id
        source_id = hashlib.md5(source_name.encode()).hexdigest()
        
        try:
            # Check if source exists and if it needs migration
            existing_source = self.sources_col.find_one({"source_name": source_name})
            
            if existing_source:
                alignment_counts = existing_source.get("alignment_counts", {})
                
                # Check if we need to migrate from lowercase to title case
                needs_migration = any(key in alignment_counts for key in ["left", "center", "right"])
                
                if needs_migration:
                    # Create new alignment counts with title case keys
                    new_counts = {
                        "Left": alignment_counts.get("left", 0) + (1 if alignment == "left" else 0),
                        "Center": alignment_counts.get("center", 0) + (1 if alignment == "center" else 0),
                        "Right": alignment_counts.get("right", 0) + (1 if alignment == "right" else 0)
                    }
                    
                    # Update the document with migrated counts
                    self.sources_col.update_one(
                        {"source_name": source_name},
                        {
                            "$set": {
                                "alignment_counts": new_counts,
                                "total_articles": existing_source.get("total_articles", 0) + 1,
                                "last_updated": datetime.now(timezone.utc).isoformat()
                            }
                        }
                    )
                else:
                    # Use title case key for the increment
                    self.sources_col.update_one(
                        {"source_name": source_name},
                        {
                            "$inc": {
                                "total_articles": 1,
                                f"alignment_counts.{mongodb_alignment}": 1
                            },
                            "$set": {
                                "last_updated": datetime.now(timezone.utc).isoformat()
                            }
                        }
                    )
            else:
                # Create a new source document with title case keys
                self.sources_col.insert_one({
                    "source_name": source_name,
                    "source_country": source_country,
                    "source_id": source_id,
                    "total_articles": 1,
                    "alignment_counts": {
                        "Left": 1 if alignment == "left" else 0,
                        "Center": 1 if alignment == "center" else 0,
                        "Right": 1 if alignment == "right" else 0
                    },
                    "first_seen": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                })
                    
        except Exception as e:
            print(f"Error updating source stats for {source_name}: {str(e)}")
    
    def _load_events_from_mongo(self):
        """Load events from MongoDB"""
        try:
            mongo_events = list(self.events_col.find({}, {"centroid_embedding": 0}))
            
            if mongo_events:
                print(f"Loaded {len(mongo_events)} existing events from MongoDB")
                self.events = mongo_events
            else:
                print("No existing events found in MongoDB")
                
        except Exception as e:
            print(f"Error loading events from MongoDB: {e}")
            try:
                with open("news_events.json", "r") as f:
                    self.events = json.load(f)
                    print(f"Loaded {len(self.events)} existing events from backup file")
            except FileNotFoundError:
                print("No existing events file found. Starting fresh.")
            except json.JSONDecodeError:
                print("Error parsing existing events file. Starting fresh.")
    
    def _save_event_to_mongo(self, event):
        """Save or update an event in MongoDB"""
        try:
            event_data = {k: v for k, v in event.items() if k != "centroid_embedding"}
            
            result = self.events_col.update_one(
                {"event_id": event["event_id"]},
                {"$set": event_data},
                upsert=True
            )
            
            for article in event.get("articles", []):
                article["event_id"] = event["event_id"]
                self.articles_col.update_one(
                    {"link": article["link"]},
                    {"$set": article},
                    upsert=True
                )
                self._update_source_stats(article)
            
            return result.upserted_id or result.modified_count
            
        except Exception as e:
            print(f"Error saving event to MongoDB: {e}")
            return False
    
    def process_articles(self):
        """Main method to process articles and update events"""
        try:
            with open("filtered_news_articles1.json", "r") as f:
                current_articles = json.load(f)
            print(f"Loaded {len(current_articles)} articles from file")
        except FileNotFoundError:
            print("No articles file found")
            return {"message": "No articles file found"}
        except json.JSONDecodeError:
            print("Error parsing articles file")
            return {"message": "Invalid articles file format"}

        # Get all existing articles from MongoDB
        existing_articles = list(self.articles_col.find({}))
        existing_article_urls = {a['link'] for a in existing_articles}
        
        # Process new articles if any exist
        new_articles = [a for a in current_articles if a.get('link') not in existing_article_urls]
        
        if new_articles:
            print(f"Processing {len(new_articles)} new articles")
            
            texts = [self._get_article_text(a) for a in new_articles]
            embeddings = self.similarity_model.encode(texts)
            
            assigned_count = 0
            unassigned_articles = []
            
            for i, (article, embedding) in enumerate(zip(new_articles, embeddings)):
                event_id = self._find_matching_event(embedding)
                if event_id is not None:
                    self._assign_article_to_event(article, event_id)
                    assigned_count += 1
                else:
                    unassigned_articles.append((article, embedding))
            
            new_event_count = 0
            if unassigned_articles:
                new_event_count = self._cluster_unassigned_articles(unassigned_articles)
            
            self._generate_political_summaries()
            self._save_results()
            
            return {
                "total_articles": len(new_articles),
                "assigned_to_existing": assigned_count,
                "created_new_events": new_event_count
            }
        else:
            print("No new articles to process - updating source information for existing articles")
            
            # Run migration first to ensure all sources have title case keys
            self._migrate_alignment_counts()
            
            # First ensure all sources exist with proper structure
            self._initialize_sources_collection()
            
            # Then update source stats
            sources_updated = 0
            for article in existing_articles:
                try:
                    if all(key in article for key in ['source', 'alignment']):
                        self._update_source_stats(article)
                        sources_updated += 1
                except Exception as e:
                    print(f"Error updating source stats for article {article.get('link')}: {e}")
            
            print(f"Updated source information for {sources_updated} existing articles")
            self._generate_political_summaries()
            self._save_results()
            
            return {
                "message": "No new articles processed, but updated source information",
                "sources_updated": sources_updated,
                "existing_articles": len(existing_articles)
            }

    def _initialize_sources_collection(self):
        """Ensure all sources have proper document structure"""
        # Get all unique sources from articles
        pipeline = [
            {"$group": {
                "_id": "$source",
                "count": {"$sum": 1},
                "source_country": {"$first": "$source_country"}
            }}
        ]
        
        sources = list(self.articles_col.aggregate(pipeline))
        
        for source in sources:
            source_name = source['_id']
            if not source_name:
                continue
                
            source_id = hashlib.md5(source_name.encode()).hexdigest()
            
            # Initialize source document if it doesn't exist
            self.sources_col.update_one(
                {"source_name": source_name},
                {
                    "$setOnInsert": {
                        "source_name": source_name,
                        "source_country": source.get("source_country", "Unknown"),
                        "source_id": source_id,
                        "total_articles": 0,
                        "alignment_counts": {
                            "Left": 0,
                            "Center": 0,
                            "Right": 0
                        },
                        "first_seen": datetime.now(timezone.utc).isoformat(),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                },
                upsert=True
            )
    
    def _find_matching_event(self, embedding: np.ndarray) -> str:
        """Find existing event that matches the article embedding"""
        max_similarity = 0
        best_event_id = None
        
        for event in self.events:
            if "centroid_embedding" not in event:
                continue
                
            centroid = np.array(event["centroid_embedding"])
            similarity = np.dot(embedding, centroid)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_event_id = event["event_id"]
        
        if max_similarity > 0:
            print(f"Best event similarity: {max_similarity:.2f} for event {best_event_id}")
        
        return best_event_id if max_similarity > SIMILARITY_THRESHOLD else None
    
    def _assign_article_to_event(self, article: dict, event_id: str):
        """Assign article to existing event"""
        event = next((e for e in self.events if e["event_id"] == event_id), None)
        if not event:
            return
        
        bias_label = self._determine_bias(article)
        
        article_data = {
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "imgURL": article.get("imgURL", ""),
            "link": article.get("link", ""),
            "timestamp": article.get("timestamp", ""),
            "location": article.get("location", ""),
            "source": article.get("source", ""),
            "source_country": article.get("source_country", "Unknown"),
            "alignment": bias_label
        }
        
        if "articles" not in event:
            event["articles"] = []
        event["articles"].append(article_data)
        
        if bias_label == "left":
            event["lCount"] = event.get("lCount", 0) + 1
        elif bias_label == "center":
            event["cCount"] = event.get("cCount", 0) + 1
        elif bias_label == "right":
            event["rCount"] = event.get("rCount", 0) + 1
        
        event["totalArticles"] = len(event["articles"])
        event["updatedAt"] = datetime.now(timezone.utc).isoformat()
        
        self._save_event_to_mongo(event)
    
    def _cluster_unassigned_articles(self, articles_with_embeddings: list) -> int:
        """Cluster unassigned articles into new events"""
        if not articles_with_embeddings:
            return 0
            
        articles = [a for a, _ in articles_with_embeddings]
        embeddings = [e for _, e in articles_with_embeddings]
        
        if len(articles) == 1:
            self._create_new_event(articles, embeddings)
            return 1
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.8,
            linkage='average'
        )
        
        try:
            clusters = clustering.fit_predict(embeddings)
        except Exception as e:
            print(f"Clustering error: {e}")
            clusters = list(range(len(articles)))
        
        unique_clusters = set(clusters)
        
        for cluster_id in unique_clusters:
            cluster_articles = [articles[i] for i, c in enumerate(clusters) if c == cluster_id]
            cluster_embeddings = [embeddings[i] for i, c in enumerate(clusters) if c == cluster_id]
            self._create_new_event(cluster_articles, cluster_embeddings)
            
        return len(unique_clusters)
    
    def _create_new_event(self, articles: list, embeddings: list):
        """Create a new event from a cluster of articles"""
        event_name = self._generate_event_name_with_gemini(articles)
        if not event_name:
            event_name = self._generate_event_name(articles)
        
        event_summary = self._generate_event_summary_with_gemini(articles)
        
        centroid = np.mean(embeddings, axis=0).tolist()
        
        bias_dist = {"left": 0, "center": 0, "right": 0}
        article_objects = []
        
        for article in articles:
            bias_label = self._determine_bias(article)
            
            if bias_label in bias_dist:
                bias_dist[bias_label] += 1
            
            article_data = {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "imgURL": article.get("imgURL", ""),
                "link": article.get("link", ""),
                "timestamp": article.get("timestamp", ""),
                "location": article.get("location", ""),
                "source": article.get("source", ""),
                "source_country": article.get("source_country", "Unknown"),
                "alignment": bias_label
            }
            article_objects.append(article_data)
        
        event_data = {
            "event_id": f"event_{len(self.events) + 1}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "eventHeadline": event_name,
            "summary": event_summary,
            "location": articles[0].get("location", "Unknown"),
            "leftSummary": "",
            "centerSummary": "",
            "rightSummary": "",
            "lCount": bias_dist["left"],
            "cCount": bias_dist["center"],
            "rCount": bias_dist["right"],
            "totalArticles": len(articles),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
            "centroid_embedding": centroid,
            "articles": article_objects
        }
        
        self.events.append(event_data)
        self._save_event_to_mongo(event_data)
    
    def _generate_political_summaries(self):
        """Generate left/center/right summaries for each event"""
        for event in self.events:
            left_articles = [a for a in event["articles"] if a.get("alignment") == "left"]
            center_articles = [a for a in event["articles"] if a.get("alignment") == "center"]
            right_articles = [a for a in event["articles"] if a.get("alignment") == "right"]
            
            if left_articles:
                event["leftSummary"] = self._generate_political_summary(left_articles, "left")
            if center_articles:
                event["centerSummary"] = self._generate_political_summary(center_articles, "center")
            if right_articles:
                event["rightSummary"] = self._generate_political_summary(right_articles, "right")
            
            self._save_event_to_mongo(event)
    
    def _generate_political_summary(self, articles: list, perspective: str) -> str:
        """Generate summary from a specific political perspective including explanation of its leaning"""
        cache_key = f"summary:{perspective}:{hash(tuple(a['title'] for a in articles[:3]))}"
        
        if cache_key in self.gemini_cache:
            return self.gemini_cache[cache_key]
            
        try:
            limiter.wait()
            titles = [a.get("title", "") for a in articles[:3]]
            prompt = f"""Write a 2-3 sentence summary of this news event from a {perspective}-leaning perspective, 
            based on these article headlines:
            
            {chr(10).join(titles)}
            
            Your response should have two parts:
        1. First, provide a 2-3 sentence summary of the event from the {perspective} perspective.
        2. Then, explain in 1-2 sentences why this summary is considered {perspective}-leaning, 
           pointing to specific language, framing, or emphasis in the headlines that reflects 
           this political perspective.
        
        Format your response like this:
        [Summary]: [summary text here]
        [Reasoning]: [reasoning text here]"""
            
            model = genai.GenerativeModel(EVENT_MODEL)
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                summary = response.text.strip()
                self.gemini_cache[cache_key] = summary
                return summary
            return ""
            
        except Exception as e:
            print(f"Error generating {perspective} summary: {e}")
            return ""
    
    def _generate_event_name_with_gemini(self, articles):
        """Generate event name with Gemini"""
        try:
            limiter.wait()
            print(f"Attempting to generate name with Gemini for event with {len(articles)} articles...")
            
            titles = [a.get('title', '') for a in articles if 'title' in a and a['title']]
            
            if not titles:
                print("No titles found in articles, skipping Gemini name generation")
                return None
                
            cache_key = f"name:{hash(tuple(titles[:2]))}"
            if cache_key in self.gemini_cache:
                return self.gemini_cache[cache_key]
                
            prompt = "Generate a short 3-5 word headline that summarizes these news articles. ONLY provide the headline text:\n\n" + "\n".join(titles[:2])
            
            print("Calling Gemini API for event name...")
            model = genai.GenerativeModel(EVENT_MODEL)
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                event_name = response.text.strip().strip('"\'').strip()
                print(f"Gemini successfully generated event name: '{event_name}'")
                if len(event_name) > 50:
                    event_name = event_name[:47] + "..."
                self.gemini_cache[cache_key] = event_name
                return event_name
            else:
                print("Gemini returned empty response for event name")
                return None
            
        except Exception as e:
            print(f"⚠️ Gemini error generating event name: {str(e)}")
            return None
    
    def _generate_event_summary_with_gemini(self, articles):
        """Generate neutral event summary with Gemini"""
        try:
            limiter.wait()
            print(f"Attempting to generate summary with Gemini for event with {len(articles)} articles...")
            
            titles_and_descriptions = []
            for article in articles[:2]:
                title = article.get('title', '')
                desc = article.get('description', '')
                if title or desc:
                    titles_and_descriptions.append(f"Title: {title}\nDescription: {desc}")
            
            if not titles_and_descriptions:
                print("No content found for summary generation")
                return "No summary available."
                
            cache_key = f"summary:{hash(tuple(titles_and_descriptions))}"
            if cache_key in self.gemini_cache:
                return self.gemini_cache[cache_key]
                
            prompt = "Write a 2-3 sentence neutral summary of this news event:\n\n" + "\n".join(titles_and_descriptions)
            
            print("Calling Gemini API for event summary...")
            model = genai.GenerativeModel(EVENT_MODEL)
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text') and response.text:
                summary = response.text.strip()
                print(f"Gemini successfully generated summary of length {len(summary)}")
                if len(summary) > 500:
                    summary = summary[:497] + "..."
                self.gemini_cache[cache_key] = summary
                return summary
            else:
                print("Gemini returned empty response for summary")
                return "No summary available."
            
        except Exception as e:
            print(f"⚠️ Gemini error generating summary: {str(e)}")
            if articles and 'description' in articles[0] and articles[0]['description']:
                print("Using article description as fallback summary")
                return articles[0]['description']
            return "No summary available."
    
    def _save_results(self):
        """Save events to JSON file without embeddings"""
        clean_events = []
        for event in self.events:
            clean_event = {k: v for k, v in event.items() if k != "centroid_embedding"}
            clean_events.append(clean_event)
        
        with open("news_events.json", "w") as f:
            # Use json_util.dumps to properly handle MongoDB types like ObjectId
            f.write(json_util.dumps(clean_events, indent=4))
        
        print(f"Saved {len(clean_events)} events to news_events.json")
    
    def _determine_bias(self, article: dict) -> str:
        """Determine article bias using BERT model"""
        text = self._get_article_text_truncated(article)
        try:
            prediction = self.bias_pipeline(text)
            label = prediction[0]['label'].upper()
            confidence = float(prediction[0]['score'])
            return label.lower() if confidence >= 0.7 else "center"
        except Exception as e:
            print(f"Error analyzing bias: {e}")
            return "center"
    
    def _get_article_text(self, article: dict) -> str:
        return f"{article.get('title', '')} {article.get('description', '')}"
    
    def _get_article_text_truncated(self, article: dict) -> str:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        words = text.split()
        return ' '.join(words[:250]) if len(words) > 250 else text
    
    def _generate_event_name(self, articles: list) -> str:
        titles = [a.get('title', '') for a in articles]
        words = ' '.join(titles).lower().split()
        common_words = [word for word in words if len(word) > 4 and word not in ['about', 'after', 'their']]
        return ' '.join(sorted(set(common_words), key=lambda x: -words.count(x))[:3]).title() if common_words else "Current Event"

    def _migrate_alignment_counts(self):
        """One-time migration to convert all lowercase alignment keys to title case"""
        print("Starting migration of alignment counts from lowercase to title case...")
        
        # Find all sources that need migration
        sources_to_migrate = list(self.sources_col.find({
            "$or": [
                {"alignment_counts.left": {"$exists": True}},
                {"alignment_counts.center": {"$exists": True}},
                {"alignment_counts.right": {"$exists": True}}
            ]
        }))
        
        migrated_count = 0
        for source in sources_to_migrate:
            try:
                counts = source.get("alignment_counts", {})
                
                # Create new counts with title case keys
                new_counts = {
                    "Left": counts.get("left", 0),
                    "Center": counts.get("center", 0),
                    "Right": counts.get("right", 0)
                }
                
                # Update the source with the new counts and remove old keys
                self.sources_col.update_one(
                    {"_id": source["_id"]},
                    {
                        "$set": {"alignment_counts": new_counts},
                        "$unset": {
                            "alignment_counts.left": "",
                            "alignment_counts.center": "",
                            "alignment_counts.right": ""
                        }
                    }
                )
                migrated_count += 1
                
            except Exception as e:
                print(f"Error migrating source {source.get('source_name')}: {e}")
        
        print(f"Migration completed. {migrated_count}/{len(sources_to_migrate)} sources migrated.")

    def __del__(self):
        """Clean up MongoDB connection when object is destroyed"""
        if hasattr(self, 'mongo_client') and self.mongo_client:
            try:
                self.mongo_client.close()
                print("MongoDB connection closed.")
            except:
                pass

if __name__ == "__main__":
    clusterer = EventClusterer()
    result = clusterer.process_articles()
    print("\nProcessing complete. Results:")
    
    if 'message' in result:
        print(f"- {result['message']}")
    else:
        print(f"- Total articles processed: {result['total_articles']}")
        print(f"- Articles assigned to existing events: {result['assigned_to_existing']}")
        print(f"- New events created: {result['created_new_events']}")
