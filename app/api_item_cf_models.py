# api_item_cf_models.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from src.item_cf import recommend_similar_items
import os

# -----------------------------
# Request body
# -----------------------------
class BookRequest(BaseModel):
    book_title: str
    top_k: int = 10

# -----------------------------
# Load pickled files from models folder
# -----------------------------
MODEL_DIR = "models"
try:
    with open(os.path.join(MODEL_DIR, "item_sim_matrix.pkl"), "rb") as f:
        item_sim_matrix = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "item_encoder.pkl"), "rb") as f:
        item_encoder = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Missing model file: {e}")

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Item-Based CF API (Pickle in models/)")

@app.get("/")
def root():
    return {"message": "API is up and running!"}

# -----------------------------
# Endpoint for book recommendations
# -----------------------------
@app.post("/recommend_items/")
def recommend_book(request: BookRequest):
    book_title = request.book_title
    top_k = request.top_k

    try:
        recommendations = recommend_similar_items(
            item_title=book_title,
            item_encoder=item_encoder,
            item_sim_matrix=item_sim_matrix,
            k=top_k
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Book '{book_title}' not found in dataset."
        )

    return {"book_title": book_title, "recommendations": list(recommendations)}

# -----------------------------
# GET endpoint for browser-friendly recommendations
# -----------------------------
@app.get("/recommend_books/")
def recommend_books_get(book_title: str, top_k: int = 10):
    """
    Recommend top_k similar books given a book title.
    Example:
    http://127.0.0.1:8000/recommend_books/?book_title=1984&top_k=10
    """
    try:
        recommendations = recommend_similar_items(
            item_title=book_title,
            item_encoder=item_encoder,
            item_sim_matrix=item_sim_matrix,
            k=top_k
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Book '{book_title}' not found in dataset."
        )

    return {"book_title": book_title, "recommendations": list(recommendations)}
