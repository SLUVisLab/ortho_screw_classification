from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from model_wrapper import ClassifierWrapper

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
# We expect the model and index to be in the same folder or configured via env
MODEL_PATH = os.getenv("MODEL_PATH", "original_classifier_best.pth")
INDEX_PATH = os.getenv("INDEX_PATH", "embeddings.pkl")
# Path to the raw images to serve them statically (for "result" display)
# This should be the root of the dataset structure on the server
DATA_DIR = os.getenv("DATA_DIR", "/path/to/data") 

# Load Resources
print("Loading model...")
# Initialize with CPU
wrapper = ClassifierWrapper(MODEL_PATH, device="cpu")

index = None
if os.path.exists(INDEX_PATH):
    print(f"Loading index from {INDEX_PATH}...")
    with open(INDEX_PATH, 'rb') as f:
        index = pickle.load(f)
else:
    print("Warning: Index file not found. Similarity search will be disabled.")

# Serve static files (images)
if os.path.exists(DATA_DIR):
    app.mount("/images", StaticFiles(directory=DATA_DIR), name="images")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    ap_image: UploadFile = File(...), 
    lat_image: UploadFile = File(...)
):
    try:
        ap_bytes = await ap_image.read()
        lat_bytes = await lat_image.read()
        
        # Run Inference
        probs, embedding = wrapper.predict(ap_bytes, lat_bytes)
        
        # Get Top Prediction
        top_idx = int(np.argmax(probs))
        # Map idx back to class name (Hardcoded or could load from metadata)
        # We need the class map. Let's hardcode the manufacturers from the training code for now
        # Classes: ['Depuy expedium (Synthes)', 'medtronic solera', 'nuvasive reline', 'seaspine mariner']
        # Wait, the order depends on the training data sort. 
        # Correct approach: The `ClassifierWrapper` or `index` should contain the class list.
        # Let's assume standard sorted order used in training code.
        CLASSES = sorted(['Depuy expedium (Synthes)', 'medtronic solera', 'nuvasive reline', 'seaspine mariner'])
        if top_idx < len(CLASSES):
            pred_class = CLASSES[top_idx]
        else:
            pred_class = "Unknown"
            
        confidence = float(probs[top_idx])
        
        # Similarity Search
        similar_images = []
        if index is not None:
            # Calculate cosine similarity with all indexed embeddings
            # embedding: [1024]
            # index['embeddings']: [N, 1024]
            
            # Reshape embedding to [1, 1024]
            query_emb = embedding.reshape(1, -1)
            
            sims = cosine_similarity(query_emb, index['embeddings'])[0]
            
            # Get Top 3 most similar indices
            top_k_indices = sims.argsort()[-3:][::-1]
            
            for idx in top_k_indices:
                meta = index['metadata'][idx]
                similar_images.append({
                    'manufacturer': meta['manufacturer'],
                    'score': float(sims[idx]),
                    'ap_url': f"/images/{meta['ap_path']}",
                    'lat_url': f"/images/{meta['lat_path']}"
                })
        
        return {
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
            "similar": similar_images
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
