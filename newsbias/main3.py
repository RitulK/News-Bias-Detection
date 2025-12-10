#this code works fine
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime, timezone
import google.generativeai as genai
from time import sleep
from collections import defaultdict

# Configure Gemini API
GOOGLE_API_KEY = "Google_API_Key"
genai.configure(api_key=GOOGLE_API_KEY)
EVENT_MODEL = "gemini-2.0-flash"

# RateLimiter class for Gemini API (similar to sample code)
class RateLimiter:
    def __init__(self, max_calls=12):  # Reduced from 15 to stay under quota
        self.calls = []
        self.max_calls = max_calls
    
    def wait(self):
        now = datetime.now(timezone.utc)
        # Remove calls older than 1 minute
        self.calls = [call for call in self.calls if (now - call).total_seconds() < 60]
        
        if len(self.calls) >= self.max_calls:
            oldest = self.calls[0]
            wait_time = (60 - (now - oldest).total_seconds()) + 0.1
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
            sleep(wait_time)
        
        self.calls.append(now)

limiter = RateLimiter()

# Initialize models
SIMILARITY_THRESHOLD = 0.65  # Lower threshold for better matching
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
        self.gemini_cache = {}  # Cache for Gemini responses
        
        try:
            with open("news_events.json", "r") as f:
                self.events = json.load(f)
                print(f"Loaded {len(self.events)} existing events")
        except FileNotFoundError:
            print("No existing events file found. Starting fresh.")
        except json.JSONDecodeError:
            print("Error parsing existing events file. Starting fresh.")
    
    def process_articles(self):
        """Main method to process articles and update events"""
        try:
            with open("filtered_news_articles1.json", "r") as f:
                current_articles = json.load(f)
        except FileNotFoundError:
            print("No articles file found")
            return {"message": "No new articles to process"}
        
        existing_article_urls = set()
        for event in self.events:
            for article in event.get("articles", []):
                if "link" in article:
                    existing_article_urls.add(article["link"])
        
        new_articles = [a for a in current_articles if a.get("link") not in existing_article_urls]
        
        if not new_articles:
            return {"message": "No new articles to process"}
        
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
            distance_threshold=0.8,  # Increased from 0.5
            linkage='average'  # Changed from 'ward'
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
            "centroid_embedding": centroid,  # Used internally only
            "articles": article_objects
        }
        
        self.events.append(event_data)
    
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
    
    def _generate_political_summary(self, articles: list, perspective: str) -> str:
        """Generate summary from a specific political perspective"""
        cache_key = f"summary:{perspective}:{hash(tuple(a['title'] for a in articles[:3]))}"
        
        if cache_key in self.gemini_cache:
            return self.gemini_cache[cache_key]
            
        try:
            limiter.wait()
            titles = [a.get("title", "") for a in articles[:3]]  # Use first 3 articles
            prompt = f"""Write a 2-3 sentence summary of this news event from a {perspective}-leaning perspective, 
            based on these article headlines:
            
            {chr(10).join(titles)}
            
            Respond with just the summary text, nothing else."""
            
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
        """Generate event name with Gemini (similar to sample code)"""
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
        """Generate neutral event summary with Gemini (similar to sample code)"""
        try:
            limiter.wait()
            print(f"Attempting to generate summary with Gemini for event with {len(articles)} articles...")
            
            titles_and_descriptions = []
            for article in articles[:2]:  # Use first 2 articles
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
            json.dump(clean_events, f, indent=4)
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

if __name__ == "__main__":
    clusterer = EventClusterer()
    result = clusterer.process_articles()
    print("\nProcessing complete. Results:")
    print(f"- Total articles processed: {result['total_articles']}")
    print(f"- Articles assigned to existing events: {result['assigned_to_existing']}")
    print(f"- New events created: {result['created_new_events']}")
