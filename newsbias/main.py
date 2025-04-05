
# from fastapi import FastAPI
# from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# import json
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from fastapi.responses import JSONResponse
# from datetime import datetime

# app = FastAPI()

# # Load models
# print("🚀 Loading models...")
# similarity_model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")
# bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
# bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)

# # Load news articles
# try:
#     with open("filtered_news_articles.json", "r", encoding="utf-8") as f:
#         news_articles = json.load(f)
#     print(f"📄 Loaded {len(news_articles)} articles.")
# except Exception as e:
#     print(f"❌ Error loading JSON file: {e}")
#     news_articles = []

# def format_published_date(date_str):
#     """Convert various date formats to consistent format"""
#     if not date_str:
#         return None
#     try:
#         dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
#         return dt.isoformat()
#     except ValueError:
#         return date_str

# def analyze_bias(text):
#     """Analyze bias with confidence threshold"""
#     prediction = bias_pipeline(text)
#     label = prediction[0]['label'].upper()  # Ensure uppercase for consistency
#     confidence = float(prediction[0]['score'])
    
#     if confidence < 0.7:
#         return "UNKNOWN", confidence
#     return label, confidence

# @app.get("/events/")
# def get_events_with_bias():
#     """Single endpoint that clusters news into events with full article details"""
#     # Initialize bias stats with uppercase keys
#     bias_stats = {"LEFT": 0, "CENTER": 0, "RIGHT": 0, "UNKNOWN": 0}
    
#     # Generate embeddings for clustering
#     texts = [
#         f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
#         for article in news_articles
#     ]
#     embeddings = similarity_model.encode(texts)
    
#     # Cluster articles
#     n_clusters = max(2, len(news_articles) // 10)
#     clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
#     labels = clustering.fit_predict(embeddings)
    
#     # Group articles into events
#     events = {}
#     for idx, (label, article) in enumerate(zip(labels, news_articles)):
#         # Get bias analysis
#         text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
#         bias_label, confidence = analyze_bias(text)
#         bias_stats[bias_label] += 1
        
#         # Create article with all details
#         analyzed_article = {
#             "title": article.get("title"),
#             "link": article.get("link"),
#             "description": article.get("description"),
#             "published_date": format_published_date(article.get("published_date")),
#             "source": article.get("source"),
#             "source_country": article.get("source_country"),
#             "news_type": article.get("news_type"),
#             "countries_mentioned": article.get("countries_mentioned", []),
#             "content": article.get("content"),
#             "bias": bias_label.capitalize(),  # Display as capitalized
#             "confidence": confidence
#         }
        
#         # Add to event cluster
#         if label not in events:
#             events[label] = []
#         events[label].append(analyzed_article)
    
#     print("\nBias Distribution:")
#     for label, count in bias_stats.items():
#         print(f"{label}: {count} articles ({count/len(news_articles)*100:.1f}%)")
    
#     # Convert to list format
#     event_list = [{"event_id": int(label), "articles": articles} for label, articles in events.items()]
    
#     return {
#         "events": event_list,
#         "bias_stats": {k.lower(): v for k, v in bias_stats.items()}  # Return lowercase for consistency
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server on http://127.0.0.1:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)






# this code works but is not the final version as it is used to test on postman.

# from fastapi import FastAPI
# from sentence_transformers import SentenceTransformer
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
# import json
# import numpy as np
# from sklearn.cluster import AgglomerativeClustering
# from fastapi.responses import JSONResponse
# from datetime import datetime
# from collections import defaultdict

# app = FastAPI()

# # Load models
# print("🚀 Loading models...")
# similarity_model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")
# bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
# bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)

# # Load news articles
# try:
#     with open("filtered_news_articles.json", "r", encoding="utf-8") as f:
#         news_articles = json.load(f)
#     print(f"📄 Loaded {len(news_articles)} articles.")
# except Exception as e:
#     print(f"❌ Error loading JSON file: {e}")
#     news_articles = []

# def format_published_date(date_str):
#     """Convert various date formats to consistent format"""
#     if not date_str:
#         return None
#     try:
#         dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
#         return dt.isoformat()
#     except ValueError:
#         return date_str

# def analyze_bias(text):
#     """Analyze bias with confidence threshold"""
#     prediction = bias_pipeline(text)
#     label = prediction[0]['label'].upper()
#     confidence = float(prediction[0]['score'])
#     return label if confidence >= 0.7 else None, confidence  # Return None for unknown

# def generate_event_name(articles):
#     """Generate event name from most common keywords in titles"""
#     titles = [article.get('title', '') for article in articles]
#     words = ' '.join(titles).lower().split()
#     common_words = [word for word in words if len(word) > 4 and word not in ['about', 'after', 'their']]
#     if not common_words:
#         return "Current Event"
#     return ' '.join(sorted(set(common_words), key=lambda x: -common_words.count(x))[:3]).title()

# def generate_summary(articles, bias_type):
#     """Generate summary for specific bias type"""
#     relevant = [a for a in articles if a['bias'].upper() == bias_type]
#     if not relevant:
#         return None
    
#     # Combine first sentences from each relevant article
#     summaries = []
#     for article in relevant[:3]:  # Use up to 3 articles
#         content = article.get('content', '')
#         first_sentence = content.split('.')[0] + '.' if '.' in content else content[:100] + '...'
#         summaries.append(first_sentence)
    
#     return ' '.join(summaries)

# @app.get("/events/")
# def get_events_with_bias():
#     """Endpoint that clusters news into named events with bias analysis"""
#     # Generate embeddings
#     texts = [f"{a.get('title', '')} {a.get('description', '')} {a.get('content', '')}" 
#              for a in news_articles]
#     embeddings = similarity_model.encode(texts)
    
#     # Cluster articles
#     n_clusters = max(2, len(news_articles) // 10)
#     clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
#     labels = clustering.fit_predict(embeddings)
    
#     # Process events
#     events = []
    
#     for event_id in set(labels):
#         event_articles = [news_articles[i] for i, lbl in enumerate(labels) if lbl == event_id]
        
#         # Process articles (filter out unknown bias)
#         processed_articles = []
#         event_bias_stats = defaultdict(int)
        
#         for article in event_articles:
#             text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
#             bias_label, confidence = analyze_bias(text)
            
#             if bias_label is None:  # Skip unknown bias articles
#                 continue
                
#             processed = {
#                 "title": article.get("title"),
#                 "link": article.get("link"),
#                 "description": article.get("description"),
#                 "published_date": format_published_date(article.get("published_date")),
#                 "source": article.get("source"),
#                 "source_country": article.get("source_country"),
#                 "content": article.get("content"),
#                 "bias": bias_label.capitalize(),
#                 "confidence": confidence
#             }
            
#             processed_articles.append(processed)
#             event_bias_stats[bias_label] += 1
        
#         # Skip events with no articles after filtering
#         if not processed_articles:
#             continue
            
#         # Generate event info
#         event_name = generate_event_name(event_articles)
        
#         event_data = {
#             "event_id": int(event_id),
#             "event_name": event_name,
#             "bias_distribution": {
#                 "left": event_bias_stats.get("LEFT", 0),
#                 "center": event_bias_stats.get("CENTER", 0),
#                 "right": event_bias_stats.get("RIGHT", 0)
#             },
#             "summaries": {
#                 "left": generate_summary(processed_articles, "LEFT"),
#                 "center": generate_summary(processed_articles, "CENTER"),
#                 "right": generate_summary(processed_articles, "RIGHT")
#             },
#             "articles": processed_articles
#         }
        
#         events.append(event_data)
    
#     return {
#         "events": sorted(events, key=lambda x: -len(x["articles"])),  # Sort by article count
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("Starting FastAPI server on http://127.0.0.1:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# will now be connected to MongoDB and will use the data from there instead of the json file although code for sending news articles into mainArticles is present in pipelineFilteration.py file.
from fastapi import FastAPI, APIRouter, Depends
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any
from pymongo import MongoClient
from bson import ObjectId
from models.events import Event
from models.articles import Article
from pipelineFilteration import result

# Import from config instead of defining here
from config.database import mainEvents, mainArticles

app = FastAPI()

# Configuration
SIMILARITY_THRESHOLD = 0.75  # Threshold for adding to existing event

# Load models
print("🚀 Loading models...")
similarity_model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")
bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)

class EventClusterer:
    def __init__(self):
        self.similarity_model = similarity_model
        self.bias_pipeline = bias_pipeline
    
    async def process_articles(self):
        """Main method to process articles and update events"""
        # Load current articles from MongoDB
        current_articles = [article for article in result if not article.get("eventID")]
        
        if not current_articles:
            return {"message": "No new articles to process"}
        
        # Generate embeddings
        texts = [self._get_article_text(a) for a in current_articles]
        embeddings = self.similarity_model.encode(texts)
        
        # First try to assign to existing events
        assigned_count = 0
        for i, (article, embedding) in enumerate(zip(current_articles, embeddings)):
            event_id = await self._find_matching_event(embedding)
            if event_id:
                await self._assign_article_to_event(article, event_id)
                assigned_count += 1
            # bring else here - if an article fails to get allocated to an event, new event will be created, upcoming articles will now be check with this newly created event as well.
        # Cluster remaining articles into new events
        remaining_articles = [a for i, a in enumerate(current_articles) 
                            if not mainArticles.find_one({"_id": a["_id"], "eventID": {"$exists": True}})]
        
        if remaining_articles:
            remaining_texts = [self._get_article_text(a) for a in remaining_articles]
            remaining_embeddings = self.similarity_model.encode(remaining_texts)
            await self._create_new_events(remaining_articles, remaining_embeddings)
        # check remaining articles if events can be formed within them
        # this above is not required as else is now inserted in the for loop above.
        return {
            "total_articles": len(current_articles),
            "assigned_to_existing": assigned_count,
            "created_new_events": len(remaining_articles) - assigned_count
        }
    
    async def _find_matching_event(self, embedding: np.ndarray) -> ObjectId:
        """Find existing event that matches the article embedding"""
        # Get all events with their centroid embeddings
        events = list(mainEvents.find({}, {"centroid_embedding": 1}))
        
        max_similarity = 0
        best_event_id = None
        
        for event in events:
            if "centroid_embedding" not in event:
                continue
                
            centroid = np.array(event["centroid_embedding"])
            similarity = np.dot(embedding, centroid)
            
            if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                max_similarity = similarity
                best_event_id = event["_id"]
        
        return best_event_id
    
    async def _assign_article_to_event(self, article: Dict, event_id: ObjectId):
        """Assign article to existing event and update event stats"""
        # Update article with eventID
        mainArticles.update_one(
            {"_id": article["_id"]},
            {"$set": {"eventID": str(event_id)}}
        )
        
        # Update event's article count and centroid
        event_articles = list(mainArticles.find({"eventID": str(event_id)}))
        event_embeddings = []
        
        for art in event_articles:
            if "embedding" in art:
                event_embeddings.append(np.array(art["embedding"]))
        
        if event_embeddings:
            new_centroid = np.mean(event_embeddings, axis=0).tolist()
            mainEvents.update_one(
                {"_id": event_id},
                {"$set": {
                    "centroid_embedding": new_centroid,
                    "articleCount": len(event_articles)
                }}
            )
    
    async def _create_new_events(self, articles: List[Dict], embeddings: List[np.ndarray]):
        """Cluster remaining articles into new events"""
        if len(articles) <= 1:
            # Single article becomes its own event
            clusters = [0]
        else:
            # Perform clustering
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="euclidean",
                linkage="ward",
                distance_threshold=1.0
            )
            clusters = clustering.fit_predict(embeddings)
        
        # Create events for each cluster
        for cluster_id in set(clusters):
            cluster_articles = [a for i, a in enumerate(articles) if clusters[i] == cluster_id]
            cluster_embeddings = [e for i, e in enumerate(embeddings) if clusters[i] == cluster_id]
            
            # Create new event
            event_name = self._generate_event_name(cluster_articles)
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            
            # Calculate bias distribution
            bias_dist = {"left": 0, "center": 0, "right": 0}
            for article in cluster_articles:
                bias_label, _ = self._analyze_bias(self._get_article_text(article))
                if bias_label:
                    bias_dist[bias_label.lower()] += 1
            
            # Insert new event
            event_data = {
                "name": event_name,
                "description": f"Automatically generated event about {event_name}",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
                "articleCount": len(cluster_articles),
                "biasDistribution": bias_dist,
                "centroid_embedding": centroid,
                "categories": ["general"]  # Default category
            }
            
            result = mainEvents.insert_one(event_data)
            event_id = result.inserted_id
            
            # Assign articles to this event
            for article, embedding in zip(cluster_articles, cluster_embeddings):
                mainArticles.update_one(
                    {"_id": article["_id"]},
                    {"$set": {
                        "eventID": str(event_id),
                        "embedding": embedding.tolist()
                    }}
                )
            # here analyze_bias needs to give bias label to the new articles as well.
    def _get_article_text(self, article: Dict) -> str:
        """Extract all text from an article"""
        return f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
    
    def _generate_event_name(self, articles: List[Dict]) -> str:
        """Generate event name from most common keywords in titles"""
        titles = [a.get('title', '') for a in articles]
        words = ' '.join(titles).lower().split()
        common_words = [word for word in words if len(word) > 4 and word not in ['about', 'after', 'their']]
        if not common_words:
            return "Current Event"
        return ' '.join(sorted(set(common_words), key=lambda x: -words.count(x))[:3]).title()
    
    def _analyze_bias(self, text: str):
        """Analyze bias with confidence threshold"""
        prediction = self.bias_pipeline(text)
        label = prediction[0]['label'].upper()
        confidence = float(prediction[0]['score'])
        return label if confidence >= 0.7 else None, confidence

# Initialize clusterer
clusterer = EventClusterer()

# Add clustering endpoint
@app.post("/cluster-articles")
async def cluster_articles():
    """Endpoint to trigger article clustering"""
    return await clusterer.process_articles()

# Include your existing router
from routers.events import router as events_router
app.include_router(events_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

