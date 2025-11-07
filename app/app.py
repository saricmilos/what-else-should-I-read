# app/app.py
import os
import traceback
from typing import List, Optional
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

# --- Config from env ---
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "book_item_model.pkl")
NN_FILE = os.getenv("NN_FILE", "nn_index.pkl")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "5000"))

# --- FastAPI app ---
app = FastAPI(title="Book Recommender API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Welcome to What Else Should I Read API!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response schemas
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

# Global objects
book_to_idx = {}
idx_to_book = {}
item_factors = None
nn = None
book_metadata = None
tfidf = None
meta_titles = None
meta_title_to_row = None
tfidf_matrix = None

# --- Utilities ---
def safe_load_models():
    """
    Load model artifacts from MODEL_DIR.
    Expects book_item_model.pkl to contain at least:
      - item_factors (array-like)
      - book_to_idx (dict)
      - idx_to_book (dict)
    Optionally: book_metadata (DataFrame or dict)
    """
    global book_to_idx, idx_to_book, item_factors, nn, book_metadata

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    nn_path = os.path.join(MODEL_DIR, NN_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    loaded = joblib.load(model_path)

    item_factors_local = loaded.get("item_factors")
    book_to_idx_local = loaded.get("book_to_idx", {})
    idx_to_book_local = loaded.get("idx_to_book", {})
    book_metadata_local = loaded.get("book_metadata", None)

    if item_factors_local is None:
        raise ValueError("item_factors not present in the model file.")

    # assign mappings
    book_to_idx = book_to_idx_local.copy() if isinstance(book_to_idx_local, dict) else dict(book_to_idx_local)
    idx_to_book = idx_to_book_local.copy() if isinstance(idx_to_book_local, dict) else dict(idx_to_book_local)
    globals()['book_to_idx'] = book_to_idx
    globals()['idx_to_book'] = idx_to_book

    # item_factors -> numpy array, ensure 2D
    item_factors_arr = np.asarray(item_factors_local)
    if item_factors_arr.ndim == 1:
        item_factors_arr = item_factors_arr.reshape(-1, 1)
    # normalize rows (L2) so cosine ~ dot
    try:
        item_factors_arr = normalize(item_factors_arr, axis=1)
    except Exception:
        # fallback: skip normalization if it fails
        pass
    globals()['item_factors'] = item_factors_arr

    # book_metadata: keep as DataFrame if possible, otherwise dict
    if isinstance(book_metadata_local, pd.DataFrame):
        globals()['book_metadata'] = book_metadata_local
    elif isinstance(book_metadata_local, dict):
        # keep dict as-is
        globals()['book_metadata'] = book_metadata_local
    elif book_metadata_local is None:
        globals()['book_metadata'] = None
    else:
        # try to coerce to DataFrame
        try:
            globals()['book_metadata'] = pd.DataFrame(book_metadata_local)
        except Exception:
            globals()['book_metadata'] = None

    # load or build NN index
    if os.path.exists(nn_path):
        try:
            globals()['nn'] = joblib.load(nn_path)
        except Exception:
            globals()['nn'] = None

    if globals().get('nn') is None:
        # build nearest neighbors with cosine metric
        n_neighbors = min(50, item_factors_arr.shape[0])
        nn_local = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
        nn_local.fit(item_factors_arr)
        globals()['nn'] = nn_local

def find_exact_or_fuzzy(title: str, cutoff: float = 0.6):
    """Return exact title if found; otherwise try fuzzy match and return matched title or None."""
    if title in book_to_idx:
        return title
    candidates = get_close_matches(title, list(book_to_idx.keys()), n=1, cutoff=cutoff)
    return candidates[0] if candidates else None

def recommend_cf(title: str, topn: int = 10, exclude_self: bool = True):
    """Return list of (title, score) from CF NearestNeighbors."""
    if title not in book_to_idx or item_factors is None or nn is None:
        return []
    bidx = book_to_idx[title]
    vec = item_factors[bidx].reshape(1, -1)
    k = min(topn + (1 if exclude_self else 0), item_factors.shape[0])
    dists, idxs = nn.kneighbors(vec, n_neighbors=k)
    idxs = idxs[0]
    dists = dists[0]
    recs = []
    for i, dist in zip(idxs, dists):
        if exclude_self and i == bidx:
            continue
        recs.append((idx_to_book[i], max(0.0, 1.0 - float(dist))))
        if len(recs) >= topn:
            break
    return recs

# --- Content-based TF-IDF fallback setup ---
def build_tfidf_on_metadata():
    """Build TF-IDF matrix on book metadata (title + author + publisher)."""
    global tfidf, meta_titles, meta_title_to_row, tfidf_matrix
    if book_metadata is None:
        return

    # try to produce a DataFrame 'md' with a 'book_title' column
    if isinstance(book_metadata, pd.DataFrame):
        md = book_metadata.copy()
    elif isinstance(book_metadata, dict):
        # dict keyed by title -> dict(fields)
        try:
            md = pd.DataFrame.from_dict(book_metadata, orient='index').reset_index().rename(columns={'index': 'book_title'})
        except Exception:
            return
    else:
        # cannot build TF-IDF
        return

    if 'book_title' not in md.columns:
        # try to infer title column
        md = md.reset_index().rename(columns={md.index.name or 'index': 'book_title'}).reset_index(drop=True)

    # build content
    md['book_title'] = md['book_title'].astype(str)
    md['content'] = (
        md['book_title'].fillna('') + ' ' +
        md.get('book_author', '').fillna('') + ' ' +
        md.get('publisher', '').fillna('').astype(str)
    )

    # restrict to titles present in CF index
    md = md[md['book_title'].isin(book_to_idx)].drop_duplicates('book_title').set_index('book_title')
    if md.shape[0] == 0:
        return

    meta_titles_local = list(md.index)
    meta_title_to_row_local = {t: i for i, t in enumerate(meta_titles_local)}
    tfidf_local = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    tfidf_matrix_local = tfidf_local.fit_transform(md['content'].values)

    globals()['meta_titles'] = meta_titles_local
    globals()['meta_title_to_row'] = meta_title_to_row_local
    globals()['tfidf'] = tfidf_local
    globals()['tfidf_matrix'] = tfidf_matrix_local

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
    cf = dict(recommend_cf(title, topn=200))
    con = dict(recommend_content(title, topn=200))
    candidates = set(cf.keys()) | set(con.keys())
    scored = []
    for c in candidates:
        s_cf = cf.get(c, 0.0)
        s_con = con.get(c, 0.0)
        score = alpha * s_cf + (1 - alpha) * s_con
        scored.append((c, score))
    scored = sorted(scored, key=lambda x: -x[1])[:topn]
    return scored

# Robust metadata extraction
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
        # if dict has single scalar value, return it
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
        # if md is scalar
        if md is not None:
            return {'book_author': _safe_to_str(md)}

    # if DataFrame
    if isinstance(book_metadata, pd.DataFrame):
        df = book_metadata
        row = None
        # Try index-based lookup first
        try:
            if 'book_title' not in df.columns:
                # index likely contains titles
                if title in df.index:
                    row = df.loc[title]
            else:
                rows = df[df['book_title'] == title]
                if len(rows) > 0:
                    row = rows.iloc[0]
        except Exception:
            row = None

        if row is None:
            # fallback: try where 'book_title' equals title even if column naming differs
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

        # convert to dict-like and extract fields
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

# --- Startup ---
@app.on_event("startup")
def startup():
    try:
        safe_load_models()
        build_tfidf_on_metadata()
        app.state.model_loaded = True
        print("Model loaded. #books:", len(book_to_idx))
    except Exception as e:
        app.state_model_loaded = False
        print("Failed to load model:", str(e))
        traceback.print_exc()

# --- Healthcheck ---
@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": bool(getattr(app.state, "model_loaded", False))}

# Debug endpoint (temporary)
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

# --- Recommend endpoint (fuzzy + hybrid fallback) ---
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
