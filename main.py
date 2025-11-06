from fastapi import FastAPI, Query
import pandas as pd
from surprise import dump
import os

app = FastAPI(title="Book Recommender API")

# ----------------------------
# Load SVD model safely
# ----------------------------
model_path = os.path.join("models", "svd_model.pkl")
try:
    _, svd_model = dump.load(model_path)
    print(f"SVD model loaded from {model_path}")
except FileNotFoundError:
    print(f"Error: {model_path} not found.")
    raise

# ----------------------------
# Load training data safely
# ----------------------------
train_csv_path = "train_data.csv"
try:
    train_df = pd.read_csv(train_csv_path)
    print(f"Training data loaded: {train_df.shape[0]} rows")
except FileNotFoundError:
    print(f"Error: {train_csv_path} not found.")
    raise

# ----------------------------
# Recommendation function
# ----------------------------
def recommend_books_for_user(model, user_id, train_df, n=5):
    all_books = train_df['book_title'].unique()
    rated_books = train_df.loc[train_df['user_id'] == user_id, 'book_title'].tolist()

    preds = []
    for book in all_books:
        if book not in rated_books:
            pred = model.predict(user_id, book).est
            preds.append((book, pred))
    
    # Sort predictions descending by estimated rating
    top_books = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    
    # Format for JSON
    return [{"book_title": b, "predicted_rating": round(r, 2)} for b, r in top_books]

# ----------------------------
# API endpoint
# ----------------------------
@app.get("/recommend")
def recommend(
    user_id: int = Query(..., description="User ID for recommendations"),
    n: int = Query(5, description="Number of books to recommend")
):
    if user_id not in train_df['user_id'].unique():
        return {"error": f"User {user_id} not found in training data."}
    
    recs = recommend_books_for_user(svd_model, user_id, train_df, n)
    return {"user_id": user_id, "recommendations": recs}

# ----------------------------
# Health check endpoint (optional)
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "rows_in_training_data": len(train_df)}
