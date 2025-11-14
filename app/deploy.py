# app/main.py
import os
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from contextlib import asynccontextmanager

logger = logging.getLogger("uvicorn.error")

MODEL_DIR = os.environ.get("MODEL_DIR") or os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)
ITEM_SIM_PATH = os.path.join(MODEL_DIR, "item_sim_matrix.pkl")
ITEM_ENCODER_PATH = os.path.join(MODEL_DIR, "item_encoder.pkl")

# In-memory placeholders
item_sim_matrix = None
item_encoder = None
models_loaded = False
book_titles_list = []

class BookRequest(BaseModel):
    book_title: str
    top_k: int = 10

# Lifespan handler to replace deprecated on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global item_sim_matrix, item_encoder, models_loaded, book_titles_list
    # Startup logic
    try:
        with open(ITEM_SIM_PATH, "rb") as f:
            item_sim_matrix = pickle.load(f)
        logger.info(f"Loaded item_sim_matrix from {ITEM_SIM_PATH}")
    except Exception as e:
        logger.error(f"Failed to load item_sim_matrix: {e}")
        item_sim_matrix = None

    try:
        with open(ITEM_ENCODER_PATH, "rb") as f:
            item_encoder = pickle.load(f)
        logger.info(f"Loaded item_encoder from {ITEM_ENCODER_PATH}")
        
        # Extract book titles from encoder
        if hasattr(item_encoder, 'classes_'):
            book_titles_list = sorted(item_encoder.classes_.tolist())
            logger.info(f"Extracted {len(book_titles_list)} book titles from encoder")
        else:
            logger.warning("item_encoder does not have 'classes_' attribute")
            
    except Exception as e:
        logger.error(f"Failed to load item_encoder: {e}")
        item_encoder = None

    models_loaded = item_sim_matrix is not None and item_encoder is not None
    if not models_loaded:
        logger.warning("Models not fully loaded; endpoints will return 503 until resolved.")

    yield  # control passes to FastAPI app while running

    # Shutdown logic (optional)
    logger.info("Shutting down API...")

# Initialize FastAPI with lifespan
app = FastAPI(title="Item-based CF API", lifespan=lifespan)

# Allow CORS for frontend (replace "*" with your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is up. Check /health and /ready."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"ready": models_loaded}

@app.get("/book_titles/")
def get_book_titles():
    """
    Returns all available book titles for autocomplete functionality.
    """
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded on server.")
    
    if not book_titles_list:
        raise HTTPException(status_code=500, detail="Book titles not available.")
    
    return {"book_titles": book_titles_list}

@app.post("/recommend_items/")
def recommend_book(request: BookRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded on server.")

    book_title = request.book_title
    top_k = request.top_k

    try:
        from src.item_cf import recommend_similar_items
        recommendations = recommend_similar_items(
            item_title=book_title,
            item_encoder=item_encoder,
            item_sim_matrix=item_sim_matrix,
            k=top_k
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Book '{book_title}' not found.")
    except Exception as e:
        logger.exception("Unexpected error during recommendation")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {"book_title": book_title, "recommendations": list(recommendations)}

@app.get("/recommend_books/")
def recommend_books_get(book_title: str, top_k: int = 10):
    return recommend_book(BookRequest(book_title=book_title, top_k=top_k))