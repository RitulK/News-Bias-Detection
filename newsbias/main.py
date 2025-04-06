import aiohttp
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
from pipelineFilteration import results
import requests
import pickle
import base64

SIMILARITY_THRESHOLD = 0.75

similarity_model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")
bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
bias_pipeline = pipeline("text-classification", model=bias_model, tokenizer=bias_tokenizer)
print("All models called successfully")
class EventClusterer:
    def __init__(self):
        self.similarity_model = similarity_model
        self.bias_pipeline = bias_pipeline
    
    async def process_articles(self):
        """Main method to process articles and update events"""
        # Load current articles from MongoDB
        current_articles = [article for article in results if not article.get("eventID")]
        
        if not current_articles:
            return {"message": "No new articles to process"}
        
        # Generate embeddings
        texts = [self._get_article_text(a) for a in current_articles]
        embeddings = self.similarity_model.encode(texts)
        
        # First try to assign to existing events
        assigned_count = 0
        event_created_count = 0
        for i, (article, embedding) in enumerate(zip(current_articles, embeddings)):
            event_id = await self._find_matching_event(embedding)
            if event_id:
                await self._assign_article_to_event(article, event_id, embedding)
                assigned_count += 1
            else:
                event_created_count += 1
                await self._create_new_events([article], [embedding])
            # bring else here - if an article fails to get allocated to an event, new event will be created, upcoming articles will now be check with this newly created event as well.
        # Cluster remaining articles into new event
        # check remaining articles if events can be formed within them
        # this above is not required as else is now inserted in the for loop above.
        return {
            "total_articles": len(current_articles),
            "assigned_to_existing": assigned_count,
            "created_new_events": event_created_count
        }
    
    async def _find_matching_event(self, embedding: np.ndarray) -> ObjectId:
        """Find existing event that matches the article embedding"""
        print("Finding matching event...")
        events = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8000/eventembeddings', timeout=5) as response:
                    if response.status == 200:
                        events = await response.json()
        except:
            return None
        
        max_similarity = 0
        best_event_id = None
        
        for event in events:
            if "centroid_embedding" not in event:
                continue
                
            centroid = pickle.loads(base64.b64decode(event["centroid_embedding"]))
            similarity = np.dot(embedding, centroid)
            
            if similarity > max_similarity and similarity > SIMILARITY_THRESHOLD:
                max_similarity = similarity
                best_event_id = event["_id"]
        print(f"Best event ID: {best_event_id}, Similarity: {max_similarity}")
        return best_event_id
    
    async def _assign_article_to_event(self, article: Dict, event_id: ObjectId, embedding: np.ndarray):
        """Assign article to existing event and update event stats"""
        print("Assigning article to event...")
        bias_label, confidence = self._analyze_bias(self._get_article_text(article))
        
        uploadArticle = {
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "imgURL": article.get("imgURL", ""),
            "eventName": article.get("eventName", ""),
            "eventID": str(event_id),
            "alignment": bias_label.lower() if bias_label else "unknown",
            "source": article.get("source", ""),
            "sourceID": article.get("sourceID", ""),
            "sourceLogo": article.get("sourceLogo", ""),
            "link": article.get("link", ""),
            "timestamp": article.get("timestamp", ""),
            "location": article.get("location", ""),
            "embedding": pickle.dumps(embedding.tolist()),  
        }
        print("posting article")
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    'http://localhost:8000/article',
                    json=uploadArticle,
                    timeout=aiohttp.ClientTimeout(total=5)
                )
        except:
            pass
        
        try:
            response = await aiohttp.ClientSession().get(
                f'http://localhost:8000/eventData/{event_id}',
                timeout=aiohttp.ClientTimeout(total=5)
            )
            event_articles = (await response.json()).get("articles", [])
        except:
            event_articles = []
        
        event_embeddings = []
        for art in event_articles:
            if "embedding" in art:
                event_embeddings.append(np.array(art["embedding"]))
        
        if event_embeddings:
            new_centroid = np.mean(event_embeddings, axis=0).tolist()
            try:
                update_payload = {
                    "centroid_embedding": new_centroid,
                    "articleCount": len(event_articles)
                }
                async with aiohttp.ClientSession() as session:
                    await session.put(
                        f'http://localhost:8000/event/{event_id}',
                        json=update_payload,
                        timeout=aiohttp.ClientTimeout(total=5)
                    )
            except:
                pass

    async def _create_new_events(self, articles: List[Dict], embeddings: List[np.ndarray]):
        """Cluster remaining articles into new events"""
        print("Creating new events...")
        if len(articles) <= 1:
            clusters = [0]
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="euclidean",
                linkage="ward",
                distance_threshold=1.0
            )
            clusters = clustering.fit_predict(embeddings)
        
        for cluster_id in set(clusters):
            cluster_articles = [a for i, a in enumerate(articles) if clusters[i] == cluster_id]
            cluster_embeddings = [e for i, e in enumerate(embeddings) if clusters[i] == cluster_id]
            
            event_name = self._generate_event_name(cluster_articles)
            centroid = np.mean(cluster_embeddings, axis=0).tolist()
            serialized_centroid = pickle.dumps(centroid)
            encoded_centroid = base64.b64encode(serialized_centroid).decode('utf-8')
            
            bias_dist = {"left": 0, "center": 0, "right": 0}
            for article in cluster_articles:
                bias_label, _ = self._analyze_bias(self._get_article_text(article))
                if bias_label:
                    bias_dist[bias_label.lower()] += 1
            
            event_data = {
                "eventHeadline": event_name,
                "location": cluster_articles[0].get("location", "Unknown"),
                "leftSummary": "",
                "centerSummary": "",
                "rightSummary": "",
                "lCount": bias_dist["left"],
                "cCount": bias_dist["center"],
                "rCount": bias_dist["right"],
                "totalArticles": len(cluster_articles),
                # "publishedDate": datetime.utcnow(),
                "centroid_embedding": encoded_centroid
            }
            print("posting a new event")
            try:
                async with aiohttp.ClientSession() as session:
                    response = await session.post(
                        'http://localhost:8000/event',
                        json=event_data,
                        timeout=aiohttp.ClientTimeout(total=5)
                    )
                    if response.status in (200, 201):
                        response_data = await response.json()
                        event_id = response_data.get("_id")
                        if event_id:
                            await self._assign_articles_to_event(cluster_articles, cluster_embeddings, event_id, event_name)
            except aiohttp.ClientError as e:
                print(f"Error creating event: {e}")
            except Exception as e:
                print(f"Unexpected error creating event: {e}")

    async def _assign_articles_to_event(self, articles: List[Dict], embeddings: List[np.ndarray], event_id: str, event_name: str):
        """Helper method to assign multiple articles to an event"""
        for article, embedding in zip(articles, embeddings):
            bias_label, confidence = self._analyze_bias(self._get_article_text(article))
            
            uploadArticle = {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "imgURL": article.get("imgURL", ""),
                "eventName": event_name,
                "eventID": str(event_id),
                "alignment": bias_label.lower() if bias_label else "unknown",
                "source": article.get("source", ""),
                "sourceID": article.get("sourceID", ""),
                "sourceLogo": article.get("sourceLogo", ""),
                "link": article.get("link", ""),
                "timestamp": article.get("timestamp", ""),
                "location": article.get("location", ""),
                "embedding": embedding.tolist()
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        'http://localhost:8000/article',
                        json=uploadArticle,
                        timeout=aiohttp.ClientTimeout(total=5)
                    )
            except:
                pass

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
print("Done")
# Initialize clusterer
clusterer = EventClusterer()
import asyncio
if __name__ == "__main__":
    asyncio.run(clusterer.process_articles())

