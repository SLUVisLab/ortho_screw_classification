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
        
        # Classes (Must match training order)
        CLASSES = sorted(['Depuy expedium (Synthes)', 'medtronic solera', 'nuvasive reline', 'seaspine mariner'])
        
        # Pre-calculate similarity with ALL embeddings
        sims = None
        if index is not None:
             query_emb = embedding.reshape(1, -1)
             sims = cosine_similarity(query_emb, index['embeddings'])[0]
        
        results = []
        
        for idx, cls in enumerate(CLASSES):
            confidence = float(probs[idx])
            
            # Find similar images SPECIFIC to this class
            class_similar = []
            if sims is not None:
                # Filter indices for this class
                # Optimization: Could build a lookup map once, but dataset is small enough
                class_indices = [i for i, m in enumerate(index['metadata']) if m['manufacturer'] == cls]
                
                if class_indices:
                    # Get similarities for these indices
                    class_sims = sims[class_indices]
                    
                    # Sort by similarity descending
                    sorted_subset_args = class_sims.argsort()[::-1]
                    
                    seen_patients = set()
                    
                    for subset_idx in sorted_subset_args:
                        if len(class_similar) >= 3:
                            break
                            
                        original_idx = class_indices[subset_idx]
                        meta = index['metadata'][original_idx]
                        pid = meta['patient_number']
                        
                        if pid not in seen_patients:
                            seen_patients.add(pid)
                            class_similar.append({
                                'manufacturer': meta['manufacturer'],
                                'score': float(sims[original_idx]),
                                'ap_url': f"/images/{meta['ap_path']}",
                                'lat_url': f"/images/{meta['lat_path']}"
                            })

            results.append({
                "manufacturer": cls,
                "confidence": confidence,
                "similar": class_similar
            })
            
        # Sort results by confidence descending
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "results": results
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
