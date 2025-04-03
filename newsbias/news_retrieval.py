from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# Load the Sentence Transformer model
model_path = "ibm-granite/granite-embedding-125m-english"
model = SentenceTransformer(model_path)

# Initialize FastAPI
app = FastAPI()

# Define request schema
class SimilarityRequest(BaseModel):
    queries: list[str]
    passages: list[str]

# Define response schema
class SimilarityResponse(BaseModel):
    similarity_matrix: list[list[float]]

@app.post("/compute_similarity", response_model=SimilarityResponse)
def compute_similarity(data: SimilarityRequest):
    try:
        # Encode queries and passages
        query_embeddings = model.encode(data.queries)
        passage_embeddings = model.encode(data.passages)
        
        # Calculate cosine similarity
        similarity_scores = util.cos_sim(query_embeddings, passage_embeddings).tolist()
        
        return {"similarity_matrix": similarity_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API using: uvicorn filename:app --reload
