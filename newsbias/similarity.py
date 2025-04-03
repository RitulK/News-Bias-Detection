# #will be used for clustering news into events based on similarity
# from fastapi import FastAPI
# from sentence_transformers import SentenceTransformer
# from pydantic import BaseModel
# import torch
# import torch.nn.functional as F
# import numpy as np

# app = FastAPI()

# # Load the model once when the server starts
# model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")

# class TextPair(BaseModel):
#     text1: str
#     text2: str

# @app.post("/similarity/")
# def get_similarity(data: TextPair):
#     sentences = [data.text1, data.text2]
#     embeddings = model.encode(sentences)
    
#     # Compute cosine similarity
#     similarity = F.cosine_similarity(
#         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#     ).item()

#     return {"similarity_score": similarity}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Load the model
model = SentenceTransformer("pauhidalgoo/finetuned-sts-ca-mpnet-base")

# Load news articles from JSON file
with open("filtered_news_articles.json", "r", encoding="utf-8") as f:
    news_articles = json.load(f)

# Extract relevant text for similarity calculation
texts = [
    f"{article['title']} {article['description']} {article['content']}"
    for article in news_articles
]

# Generate embeddings
embeddings = model.encode(texts)

# Determine the number of clusters (adjustable)
n_clusters = max(2, len(news_articles) // 10)

# Perform clustering using Euclidean metric (since cosine is not directly supported)
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
labels = clustering.fit_predict(embeddings)

# Convert numpy labels to Python int
labels = labels.astype(int)

# Group articles into events
events = {}
for idx, label in enumerate(labels):
    label = int(label)  # Ensure numpy.int64 is converted to standard int
    if label not in events:
        events[label] = []
    events[label].append(news_articles[idx])

@app.get("/events/")
def get_clustered_events():
    # Convert all numpy types to Python native types before returning
    converted_events = {
        int(k): jsonable_encoder(v) for k, v in events.items()
    }
    return JSONResponse(content={"events": converted_events})

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)




