# app/app.py
import os
import traceback
import asyncio
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from difflib import get_close_matches
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -------------------------
# Config from environment
# -------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "book_item_model.pkl")
NN_FILE = os.getenv("NN_FILE", "nn_index.pkl")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "5000"))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("book-recommender")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Book Recommender API", version="1.0")

# Normalize CORS origins
def _parse_cors(origins: str):
    if not origins:
        return []
    # allow single "*" to signify all origins
    if origins.strip() == "*":
        return ["*"]
    return [o.strip() for o in origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors(CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to What Else Should I Read API!"}

# -------------------------
# Response schemas
# -------------------------
class RecItem(BaseModel):
    book_title: str
    score: float
    author: Optional[str] = None
    publisher: Optional[str] = None
    year: Optional[str] = None
    image: Optional[str] = None

class RecommendResponse(BaseModel):
    query: str
    matched_title: Optional[str]
    recommendations: List[RecItem]

# -------------------------
# Global objects (initialized at startup)
# -------------------------
book_to_idx: Dict[str, int] = {}
idx_to_book: Dict[int, str] = {}
item_factors: Optional[np.ndarray] = None
nn: Optional[NearestNeighbors] = None
book_metadata: Optional[Any] = None  # DataFrame or dict
tfidf: Optional[TfidfVectorizer] = None
meta_titles: Optional[List[str]] = None
meta_title_to_row: Optional[Dict[str, int]] = None
tfidf_matrix = None

# -------------------------
# Utilities / Model loading
# -------------------------
def safe_load_models():
    """
    Load model artifacts from MODEL_DIR.
    Expects a joblib file with keys:
      - item_factors (array-like)
      - book_to_idx (dict)
      - idx_to_book (dict)
    Optionally: book_metadata (DataFrame or dict)
    """
    global book_to_idx, idx_to_book, item_factors, nn, book_metadata

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    nn_path = os.path.join(MODEL_DIR, NN_FILE)

    logger.info("Loading model from %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    loaded = joblib.load(model_path)

    item_factors_local = loaded.get("item_factors")
    book_to_idx_local = loaded.get("book_to_idx", {})
    idx_to_book_local = loaded.get("idx_to_book", {})
    book_metadata_local = loaded.get("book_metadata", None)

    if item_factors_local is None:
        raise ValueError("item_factors not present in the model file.")

    # assign mappings (make copies to avoid accidental mutation)
    if isinstance(book_to_idx_local, dict):
        book_to_idx = book_to_idx_local.copy()
    else:
        book_to_idx = dict(book_to_idx_local)

    if isinstance(idx_to_book_local, dict):
        idx_to_book = idx_to_book_local.copy()
    else:
        idx_to_book = dict(idx_to_book_local)

    # item_factors -> numpy array, ensure 2D
    item_factors_arr = np.asarray(item_factors_local)
    if item_factors_arr.ndim == 1:
        item_factors_arr = item_factors_arr.reshape(-1, 1)

    # normalize rows (L2) so cosine ~ dot
    try:
        item_factors_arr = normalize(item_factors_arr, axis=1)
    except Exception:
        logger.warning("Failed to normalize item_factors; continuing without normalization", exc_info=True)

    # book_metadata handling
    if isinstance(book_metadata_local, pd.DataFrame):
        book_metadata = book_metadata_local.copy()
    elif isinstance(book_metadata_local, dict):
        book_metadata = book_metadata_local.copy()
    elif book_metadata_local is None:
        book_metadata = None
    else:
        try:
            book_metadata = pd.DataFrame(book_metadata_local)
        except Exception:
            logger.warning("Unable to coerce book_metadata to DataFrame; storing None", exc_info=True)
            book_metadata = None

    # set globals safely
    globals()['book_to_idx'] = book_to_idx
    globals()['idx_to_book'] = idx_to_book
    globals()['item_factors'] = item_factors_arr
    globals()['book_metadata'] = book_metadata

    # load or build NN index
    if os.path.exists(nn_path):
        try:
            logger.info("Loading NN index from %s", nn_path)
            globals()['nn'] = joblib.load(nn_path)
        except Exception:
            logger.warning("Failed to load prebuilt NN index; will build a new one", exc_info=True)
            globals()['nn'] = None

    if globals().get('nn') is None:
        logger.info("Building NearestNeighbors index (metric=cosine)")
        n_neighbors = min(50, item_factors_arr.shape[0])
        nn_local = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
        nn_local.fit(item_factors_arr)
        globals()['nn'] = nn_local

def find_exact_or_fuzzy(title: str, cutoff: float = 0.6) -> Optional[str]:
    """Return exact title if found; otherwise try fuzzy match and return matched title or None."""
    if not title:
        return None
    # exact match
    if title in book_to_idx:
        return title
    # try case-insensitive exact
    lower_map = {k.lower(): k for k in book_to_idx.keys()}
    if title.lower() in lower_map:
        return lower_map[title.lower()]
    # fuzzy match
    candidates = get_close_matches(title, list(book_to_idx.keys()), n=1, cutoff=cutoff)
    return candidates[0] if candidates else None

def recommend_cf(title: str, topn: int = 10, exclude_self: bool = True):
    """Return list of (title, score) from CF NearestNeighbors."""
    if title not in book_to_idx or item_factors is None or nn is None:
        return []
    bidx = book_to_idx[title]
    vec = item_factors[bidx].reshape(1, -1)
    k = min(topn + (1 if exclude_self else 0), item_factors.shape[0])
    # kneighbors returns distances (cosine distance if metric='cosine')
    dists, idxs = nn.kneighbors(vec, n_neighbors=k)
    idxs = idxs[0]
    dists = dists[0]
    recs = []
    for i, dist in zip(idxs, dists):
        if exclude_self and i == bidx:
            continue
        # convert cosine distance to similarity-like score in [0,1]
        score = max(0.0, 1.0 - float(dist))
        recs.append((idx_to_book.get(i, ""), score))
        if len(recs) >= topn:
            break
    return recs

# -------------------------
# TF-IDF / content fallback
# -------------------------
def build_tfidf_on_metadata():
    """Build TF-IDF matrix on book metadata (title + author + publisher)."""
    global tfidf, meta_titles, meta_title_to_row, tfidf_matrix
    if book_metadata is None:
        logger.info("No book_metadata available; skipping TF-IDF build")
        return

    # normalize book_metadata into DataFrame with 'book_title' column
    if isinstance(book_metadata, pd.DataFrame):
        md = book_metadata.copy()
        # try to ensure title column exists or index holds title
        if 'book_title' not in md.columns and md.index.name is None:
            # leave as-is; we'll try index later
            pass
    elif isinstance(book_metadata, dict):
        try:
            md = pd.DataFrame.from_dict(book_metadata, orient='index').reset_index().rename(columns={'index': 'book_title'})
        except Exception:
            logger.exception("Failed to coerce dict book_metadata into DataFrame")
            return
    else:
        logger.warning("Unsupported book_metadata type for TF-IDF: %s", type(book_metadata))
        return

    # Ensure a 'book_title' field
    if 'book_title' not in md.columns:
        # if index contains titles, move to column
        md = md.reset_index().rename(columns={md.index.name or 'index': 'book_title'}).reset_index(drop=True)

    md['book_title'] = md['book_title'].astype(str)
    # Build a 'content' field combining title, author, publisher
    md['content'] = (
        md['book_title'].fillna('') + ' ' +
        md.get('book_author', pd.Series('', index=md.index)).fillna('').astype(str) + ' ' +
        md.get('publisher', pd.Series('', index=md.index)).fillna('').astype(str)
    )

    # restrict to titles present in CF index
    md = md[md['book_title'].isin(book_to_idx)].drop_duplicates('book_title').set_index('book_title')
    if md.shape[0] == 0:
        logger.info("No overlapping titles between book_metadata and CF index; skipping TF-IDF")
        return

    meta_titles_local = list(md.index)
    meta_title_to_row_local = {t: i for i, t in enumerate(meta_titles_local)}
    tfidf_local = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    tfidf_matrix_local = tfidf_local.fit_transform(md['content'].values)

    globals()['meta_titles'] = meta_titles_local
    globals()['meta_title_to_row'] = meta_title_to_row_local
    globals()['tfidf'] = tfidf_local
    globals()['tfidf_matrix'] = tfidf_matrix_local
    logger.info("Built TF-IDF matrix for %d titles", len(meta_titles_local))

def recommend_content(title: str, topn: int = 10):
    """Content-based recommendations using TF-IDF content matrix."""
    if tfidf_matrix is None or meta_title_to_row is None or title not in meta_title_to_row:
        return []
    r = meta_title_to_row[title]
    sims = cosine_similarity(tfidf_matrix[r], tfidf_matrix).ravel()
    top_idx = sims.argsort()[::-1]
    results = []
    for i in top_idx:
        t = meta_titles[i]
        if t == title:
            continue
        results.append((t, float(sims[i])))
        if len(results) >= topn:
            break
    return results

def hybrid_recommend(title: str, topn: int = 10, alpha: float = 0.6):
    """Blend CF and content recommendations (alpha weight to CF)."""
    cf_list = recommend_cf(title, topn=200)
    con_list = recommend_content(title, topn=200)
    cf = dict(cf_list)
    con = dict(con_list)
    candidates = set(cf.keys()) | set(con.keys())
    scored = []
    for c in candidates:
        s_cf = cf.get(c, 0.0)
        s_con = con.get(c, 0.0)
        score = alpha * s_cf + (1 - alpha) * s_con
        scored.append((c, score))
    scored = sorted(scored, key=lambda x: -x[1])[:topn]
    return scored

# -------------------------
# Metadata utilities
# -------------------------
def _safe_to_str(v):
    """Return simple string for scalar-like values; otherwise None."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    try:
        import numpy as _np
        if isinstance(v, _np.generic):
            return str(v)
    except Exception:
        pass
    if isinstance(v, dict):
        if len(v) == 1:
            single = next(iter(v.values()))
            if isinstance(single, (str, int, float, bool)):
                return str(single)
        return None
    try:
        import pandas as _pd
        if isinstance(v, _pd.Series):
            d = v.to_dict()
            for val in d.values():
                if isinstance(val, (str, int, float, bool)):
                    return str(val)
            return None
    except Exception:
        pass
    try:
        s = str(v)
        if "object at 0x" not in s:
            return s
    except Exception:
        pass
    return None

def get_metadata_for_title(title: str):
    """
    Return dict with keys: book_author, publisher, year_of_publication, image_url_l
    Handles many shapes of book_metadata (DataFrame, dict-of-dicts).
    """
    if book_metadata is None:
        return {}

    # if dict keyed by title
    if isinstance(book_metadata, dict):
        md = book_metadata.get(title)
        if isinstance(md, dict):
            return {
                'book_author': _safe_to_str(md.get('book_author') or md.get('author')),
                'publisher': _safe_to_str(md.get('publisher')),
                'year_of_publication': _safe_to_str(md.get('year_of_publication') or md.get('year')),
                'image_url_l': _safe_to_str(md.get('image_url_l') or md.get('image')),
            }
        if md is not None:
            return {'book_author': _safe_to_str(md)}

    # if DataFrame
    if isinstance(book_metadata, pd.DataFrame):
        df = book_metadata
        row = None
        # Try index-based lookup first
        try:
            if 'book_title' not in df.columns:
                if title in df.index:
                    row = df.loc[title]
            else:
                rows = df[df['book_title'] == title]
                if len(rows) > 0:
                    row = rows.iloc[0]
        except Exception:
            row = None

        if row is None:
            # fallback: try where any column with 'title' in its name equals the title
            try:
                possible_cols = [c for c in df.columns if 'title' in c.lower()]
                for c in possible_cols:
                    rows = df[df[c].astype(str) == title]
                    if len(rows) > 0:
                        row = rows.iloc[0]
                        break
            except Exception:
                row = None

        if row is None:
            return {}

        try:
            rdict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        except Exception:
            try:
                rdict = dict(row)
            except Exception:
                rdict = {}

        return {
            'book_author': _safe_to_str(rdict.get('book_author') or rdict.get('author')),
            'publisher': _safe_to_str(rdict.get('publisher')),
            'year_of_publication': _safe_to_str(rdict.get('year_of_publication') or rdict.get('year')),
            'image_url_l': _safe_to_str(rdict.get('image_url_l') or rdict.get('image')),
        }

    return {}

# -------------------------
# Startup / Shutdown events
# -------------------------
async def _initialize_all():
    """
    Helper to run the potentially-blocking initialization functions off the event loop.
    """
    # run safe_load_models and build_tfidf_on_metadata off the event loop to avoid blocking
    await asyncio.to_thread(safe_load_models)
    # build TF-IDF may use CPU; run in thread as well
    await asyncio.to_thread(build_tfidf_on_metadata)

@app.on_event("startup")
async def startup_event():
    logger.info("Startup: beginning model initialization")
    # set a default so endpoints can check quickly
    app.state.model_loaded = False
    try:
        await _initialize_all()
        app.state.model_loaded = True
        try:
            count = len(book_to_idx)
        except Exception:
            count = 0
        logger.info("Model loaded successfully. #books: %d", count)
    except Exception as exc:
        app.state.model_loaded = False
        logger.exception("Failed to initialize models during startup: %s", exc)
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutdown: cleaning up resources")
    # If you have GPU or other resources to free, do it here.
    # For example, if you stored a big model in app.state.model_ref, release it.
    # We keep this simple for now.
    await asyncio.sleep(0)

# -------------------------
# Healthcheck
# -------------------------
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "num_books_indexed": len(book_to_idx) if book_to_idx else 0
    }

# -------------------------
# Debug metadata endpoint
# -------------------------
@app.get("/_debug_metadata")
def debug_metadata(sample_title: Optional[str] = None):
    t = sample_title or (next(iter(book_to_idx.keys())) if book_to_idx else None)
    sample_value = None
    try:
        if isinstance(book_metadata, dict):
            sample_value = book_metadata.get(t)
        elif isinstance(book_metadata, pd.DataFrame):
            if 'book_title' in book_metadata.columns:
                rows = book_metadata[book_metadata['book_title'] == t]
                sample_value = rows.iloc[0].to_dict() if len(rows) > 0 else None
            elif t in book_metadata.index:
                sample_value = book_metadata.loc[t].to_dict()
    except Exception as ex:
        sample_value = str(ex)
    return {
        "type": str(type(book_metadata)),
        "sample_title": t,
        "sample_value": sample_value
    }

# -------------------------
# Recommend endpoint
# -------------------------
@app.get("/recommend", response_model=RecommendResponse)
def recommend(
    book_title: str = Query(..., description="Book title (exact or close)"),
    topn: int = Query(10, ge=1, le=50, description="Number of recommendations"),
    use_hybrid: bool = Query(True, description="Use hybrid CF+content fallback if available")
):
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(status_code=503, detail="Model not loaded")

    matched = find_exact_or_fuzzy(book_title, cutoff=0.55)
    if matched is None:
        raise HTTPException(status_code=404, detail=f"Book not found (no close match): {book_title!r}")

    recs = recommend_cf(matched, topn=topn)
    if (not recs or len(recs) < 3) and use_hybrid and tfidf_matrix is not None:
        recs = hybrid_recommend(matched, topn=topn, alpha=0.6)

    out = []
    for title, score in recs:
        meta = get_metadata_for_title(title) or {}
        out.append(RecItem(
            book_title=title,
            score=float(score),
            author=meta.get('book_author'),
            publisher=meta.get('publisher'),
            year=str(meta.get('year_of_publication')) if meta.get('year_of_publication') is not None else None,
            image=meta.get('image_url_l')
        ))

    return RecommendResponse(query=book_title, matched_title=matched, recommendations=out)

# -------------------------
# Run with uvicorn when executed directly
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
