"""
=============================================================
  Social Media Trend Prediction — Backend with REAL-TIME DATA
=============================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import re
import time
import requests as http_requests
import threading
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app  = Flask(__name__)
CORS(app)

MODEL_PATH = "trained_models.pkl"

# In-memory cache for live feed
live_feed_cache    = []
cache_last_updated = 0
CACHE_TTL          = 300   # refresh every 5 minutes

# ─────────────────────────────────────────────────────────────
# REAL-TIME REDDIT DATA FETCHER
# Uses Reddit's free public .json endpoint — NO API KEY needed
# ─────────────────────────────────────────────────────────────

TRENDING_SUBREDDITS = [
    "worldnews", "technology", "science", "artificial",
    "news", "todayilearned", "Futurology", "space"
]

NON_TRENDING_SUBREDDITS = [
    "AskReddit", "casualconversation", "mildlyinteresting",
    "Showerthoughts", "LifeProTips"
]

REDDIT_HEADERS = {
    "User-Agent": "TrendPredictor/1.0 (Research Project)"
}


def fetch_reddit_posts(subreddit: str, sort: str = "hot", limit: int = 25) -> list:
    """
    Fetch posts from a public subreddit using Reddit's free JSON API.
    No API key or authentication required.
    """
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
    try:
        resp = http_requests.get(url, headers=REDDIT_HEADERS, timeout=8)
        if resp.status_code != 200:
            return []
        data  = resp.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            p     = child.get("data", {})
            title = p.get("title", "").strip()
            if not title or len(title) < 10:
                continue
            posts.append({
                "id"          : p.get("id", ""),
                "title"       : title,
                "subreddit"   : p.get("subreddit", subreddit),
                "score"       : p.get("score", 0),
                "comments"    : p.get("num_comments", 0),
                "upvote_ratio": p.get("upvote_ratio", 0.5),
                "created_utc" : p.get("created_utc", 0),
                "url"         : "https://reddit.com" + p.get("permalink", ""),
                "flair"       : p.get("link_flair_text", "") or ""
            })
        return posts
    except Exception as e:
        print(f"  [Reddit] Failed to fetch r/{subreddit}: {e}")
        return []


def fetch_live_reddit_feed(limit_per_sub: int = 10) -> list:
    """
    Fetch real posts from multiple subreddits.
    Trending subreddits  → source_type = 'trending'
    Non-trending subs    → source_type = 'non_trending'
    """
    feed = []

    for sub in TRENDING_SUBREDDITS[:4]:
        posts = fetch_reddit_posts(sub, sort="hot", limit=limit_per_sub)
        for p in posts:
            p["source_type"] = "trending"
            p["platform"]    = f"Reddit r/{sub}"
        feed.extend(posts)
        time.sleep(0.4)   # polite rate limiting

    for sub in NON_TRENDING_SUBREDDITS[:3]:
        posts = fetch_reddit_posts(sub, sort="new", limit=limit_per_sub)
        for p in posts:
            p["source_type"] = "non_trending"
            p["platform"]    = f"Reddit r/{sub}"
        feed.extend(posts)
        time.sleep(0.4)

    return feed


def get_live_feed(force_refresh: bool = False) -> list:
    """Return cached feed or refresh if stale."""
    global live_feed_cache, cache_last_updated
    if force_refresh or (time.time() - cache_last_updated) > CACHE_TTL:
        print("  [Cache] Refreshing live Reddit feed...")
        fresh = fetch_live_reddit_feed(limit_per_sub=15)
        if fresh:
            live_feed_cache    = fresh
            cache_last_updated = time.time()
            print(f"  [Cache] Fetched {len(fresh)} live posts.")
    return live_feed_cache


# ─────────────────────────────────────────────────────────────
# SYNTHETIC FALLBACK DATASET
# Used when Reddit is unreachable or to bulk up training data
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 1200) -> tuple:
    trending_templates = [
        "Breaking: {event} happening right now across the world",
        "VIRAL: {person} {action} and the internet is going crazy",
        "JUST IN: {country} reports record {event} thousands affected",
        "{event} goes viral with over {num} million views in 24 hours",
        "Emergency alert: {event} confirmed by officials worldwide",
        "Historic: {person} becomes first to {action} ever recorded",
        "Trending worldwide: {topic} takes over social media today",
        "Live update: {event} situation escalating people watching",
        "World record broken: {person} achieves {action} astonishing",
        "#WorldCup final match tonight packed stadium millions watching",
        "Bitcoin surges to all time high {num}k investors react wildly",
        "New AI model surpasses human performance benchmark confirmed",
        "Election results announced {country} voters react to decision",
        "Major earthquake strikes {country} emergency services deployed",
        "Government announces major policy change affecting millions",
        "Space agency confirms major discovery scientists thrilled",
        "Massive protest erupts thousands take to streets globally",
        "Breakthrough vaccine approved for deadly disease worldwide",
        "Stock market crashes investors panic billions wiped out today",
        "Tech giant announces revolutionary product launch globally",
    ]
    non_trending_templates = [
        "Just had {food} for lunch it was pretty good nothing special",
        "My {pet} is sleeping on the couch again typical afternoon",
        "Weather today is {weather} might stay indoors watch a show",
        "Trying a new {hobby} class this weekend should be interesting",
        "Coffee shop near my office finally opened good wifi today",
        "Reminder to drink more water today hydration is important",
        "Reorganizing my room this weekend takes so long but worth it",
        "Reading a new book about {topic} quite interesting so far",
        "Cooked dinner at home tonight simple pasta turned out decent",
        "Taking a walk in the park this evening fresh air feels nice",
        "Finished my weekend workout routine feeling tired but good",
        "Planning to watch a movie tonight haven't decided yet",
        "Trying to learn guitar again picked it up after a long break",
        "Grocery shopping done for the week meal prep starts tomorrow",
        "Early morning have a meeting but coffee is helping me focus",
        "Watering my plants they seem to be doing well this week",
        "Just updated my phone software took forever but works now",
        "Making a to-do list for tomorrow trying to stay organized",
        "Watched a documentary about nature quite calming overall",
        "Writing in my journal before bed peaceful routine I recommend",
    ]
    events   = ["earthquake","election","summit","championship","crisis","breakthrough","launch"]
    persons  = ["scientist","leader","athlete","CEO","artist","president","activist"]
    actions  = ["sets world record","makes history","wins award","breaks record"]
    countries= ["India","USA","China","UK","Germany","Brazil","Japan","France"]
    topics   = ["AI","climate","economy","technology","science","health","space"]
    foods    = ["pasta","sushi","tacos","pizza","curry","salad","sandwich"]
    pets     = ["cat","dog","rabbit","hamster","parrot"]
    weathers = ["sunny","cloudy","rainy","windy","cold","warm","humid"]
    hobbies  = ["painting","cooking","yoga","pottery","dance","photography"]
    nums     = [5, 10, 15, 20, 50, 100]

    def fill(t):
        return (t.replace("{event}",   np.random.choice(events))
                 .replace("{person}",  np.random.choice(persons))
                 .replace("{action}",  np.random.choice(actions))
                 .replace("{country}", np.random.choice(countries))
                 .replace("{topic}",   np.random.choice(topics))
                 .replace("{food}",    np.random.choice(foods))
                 .replace("{pet}",     np.random.choice(pets))
                 .replace("{weather}", np.random.choice(weathers))
                 .replace("{hobby}",   np.random.choice(hobbies))
                 .replace("{num}",     str(np.random.choice(nums))))

    np.random.seed(42)
    half   = n_samples // 2
    texts  = [fill(np.random.choice(trending_templates))     for _ in range(half)]
    texts += [fill(np.random.choice(non_trending_templates)) for _ in range(half)]
    labels = [1] * half + [0] * half
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ─────────────────────────────────────────────────────────────
# BUILD TRAINING DATASET
# Combines real Reddit posts + synthetic data → ~1500 samples
# ─────────────────────────────────────────────────────────────

def build_training_dataset() -> tuple:
    print("\n  [Data] Fetching real Reddit posts for training...")
    reddit_posts = fetch_live_reddit_feed(limit_per_sub=20)

    texts, labels = [], []
    for p in reddit_posts:
        title = p.get("title", "").strip()
        if len(title) < 8:
            continue
        labels.append(1 if p["source_type"] == "trending" else 0)
        texts.append(title)

    reddit_count = len(texts)
    print(f"  [Data] Collected {reddit_count} real Reddit posts.")

    # Fill to ~1500 with synthetic data
    needed = max(0, 1500 - reddit_count)
    if needed > 0:
        print(f"  [Data] Adding {needed} synthetic samples...")
        syn_texts, syn_labels = generate_synthetic_dataset(n_samples=needed)
        texts  += syn_texts
        labels += syn_labels

    # Shuffle
    combined = list(zip(texts, labels))
    np.random.shuffle(combined)
    texts, labels = zip(*combined)
    print(f"  [Data] Final: {len(texts)} samples  "
          f"({sum(labels)} trending / {len(labels)-sum(labels)} non-trending)")
    return list(texts), list(labels)


# ─────────────────────────────────────────────────────────────
# ML TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def train_models():
    print("\n" + "="*62)
    print("   SOCIAL MEDIA TREND PREDICTION — ML TRAINING PIPELINE")
    print("="*62)

    # 1. Dataset
    print("\n[1/6] Building dataset (Real Reddit + Synthetic)...")
    texts, labels = build_training_dataset()

    # 2. TF-IDF Vectorization
    print("\n[2/6] Applying TF-IDF vectorization (max 5000 features, bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2
    )
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    print(f"      Feature matrix: {X.shape}")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[3/6] Split → Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    results = {}

    # 4a. Logistic Regression + GridSearchCV
    print("\n[4/6] Training Logistic Regression + GridSearchCV...")
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42),
        {"C": [0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"], "max_iter": [300]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    lr_pred = lr_grid.best_estimator_.predict(X_test)
    results["lr"] = {
        "model"      : lr_grid.best_estimator_,
        "name"       : "Logistic Regression",
        "accuracy"   : round(accuracy_score(y_test, lr_pred) * 100, 2),
        "precision"  : round(precision_score(y_test, lr_pred, zero_division=0) * 100, 2),
        "recall"     : round(recall_score(y_test, lr_pred, zero_division=0) * 100, 2),
        "f1"         : round(f1_score(y_test, lr_pred, zero_division=0) * 100, 2),
        "best_params": lr_grid.best_params_,
        "cv_score"   : round(lr_grid.best_score_ * 100, 2)
    }
    print(f"      Accuracy: {results['lr']['accuracy']}%   Params: {lr_grid.best_params_}")

    # 4b. SVM + GridSearchCV
    print("\n      Training SVM + GridSearchCV (may take ~30s)...")
    svm_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    svm_pred = svm_grid.best_estimator_.predict(X_test)
    results["svm"] = {
        "model"      : svm_grid.best_estimator_,
        "name"       : "SVM (RBF Kernel)",
        "accuracy"   : round(accuracy_score(y_test, svm_pred) * 100, 2),
        "precision"  : round(precision_score(y_test, svm_pred, zero_division=0) * 100, 2),
        "recall"     : round(recall_score(y_test, svm_pred, zero_division=0) * 100, 2),
        "f1"         : round(f1_score(y_test, svm_pred, zero_division=0) * 100, 2),
        "best_params": svm_grid.best_params_,
        "cv_score"   : round(svm_grid.best_score_ * 100, 2)
    }
    print(f"      Accuracy: {results['svm']['accuracy']}%   Params: {svm_grid.best_params_}")

    # 4c. Random Forest + GridSearchCV
    print("\n      Training Random Forest + GridSearchCV...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 20], "min_samples_split": [2, 5]},
        cv=5, scoring="accuracy", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    rf_pred = rf_grid.best_estimator_.predict(X_test)
    results["rf"] = {
        "model"      : rf_grid.best_estimator_,
        "name"       : "Random Forest",
        "accuracy"   : round(accuracy_score(y_test, rf_pred) * 100, 2),
        "precision"  : round(precision_score(y_test, rf_pred, zero_division=0) * 100, 2),
        "recall"     : round(recall_score(y_test, rf_pred, zero_division=0) * 100, 2),
        "f1"         : round(f1_score(y_test, rf_pred, zero_division=0) * 100, 2),
        "best_params": rf_grid.best_params_,
        "cv_score"   : round(rf_grid.best_score_ * 100, 2)
    }
    print(f"      Accuracy: {results['rf']['accuracy']}%   Params: {rf_grid.best_params_}")

    # 5. Pick best
    best_key = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n[5/6] Best model → {results[best_key]['name']}  ({results[best_key]['accuracy']}%)")

    # 6. Save
    print("\n[6/6] Saving to disk...")
    joblib.dump({
        "vectorizer": vectorizer,
        "models"    : results,
        "best_model": best_key,
        "label_map" : {1: "Trending", 0: "Non-Trending"}
    }, MODEL_PATH)
    print(f"      Saved → {MODEL_PATH}")

    print("\n" + "="*62)
    print("  RESULTS SUMMARY")
    print("="*62)
    for k, v in results.items():
        flag = "  ← BEST" if k == best_key else ""
        print(f"  {v['name']:28s}  Acc={v['accuracy']}%  F1={v['f1']}%{flag}")
    print("="*62 + "\n")

    return results, vectorizer, best_key


# ─────────────────────────────────────────────────────────────
# STARTUP — LOAD SAVED OR TRAIN FRESH
# ─────────────────────────────────────────────────────────────

print("\nStarting Social Media Trend Prediction API...")

if os.path.exists(MODEL_PATH):
    print(f"Loading saved models from '{MODEL_PATH}'...")
    _saved     = joblib.load(MODEL_PATH)
    VECTORIZER = _saved["vectorizer"]
    MODELS     = _saved["models"]
    BEST_MODEL = _saved["best_model"]
    LABEL_MAP  = _saved["label_map"]
    print("Models loaded successfully.")
else:
    print("No saved models found — training now (~60 seconds)...")
    MODELS, VECTORIZER, BEST_MODEL = train_models()
    LABEL_MAP = {1: "Trending", 0: "Non-Trending"}

# Warm the feed cache in background so first /live_feed call is instant
threading.Thread(target=get_live_feed, args=(True,), daemon=True).start()


# ─────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9#\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_single(text: str, model_key: str) -> dict:
    if model_key not in MODELS:
        model_key = BEST_MODEL
    clean = preprocess(text)
    X_vec = VECTORIZER.transform([clean])
    clf   = MODELS[model_key]["model"]
    label = int(clf.predict(X_vec)[0])
    proba = clf.predict_proba(X_vec)[0]
    t_prob = float(round(proba[1], 4))
    n_prob = float(round(proba[0], 4))
    return {
        "prediction"        : LABEL_MAP[label],
        "confidence"        : float(round(t_prob if label == 1 else n_prob, 4)),
        "trending_prob"     : t_prob,
        "non_trending_prob" : n_prob,
        "model_used"        : MODELS[model_key]["name"],
        "model_key"         : model_key,
        "model_accuracy"    : MODELS[model_key]["accuracy"],
        "preprocessed_text" : clean,
        "token_count"       : len(clean.split()),
        "is_best_model"     : (model_key == BEST_MODEL)
    }


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — frontend pings this on load."""
    return jsonify({
        "status"       : "ok",
        "message"      : "Social Media Trend Prediction API is running",
        "models_loaded": list(MODELS.keys()),
        "best_model"   : BEST_MODEL,
        "timestamp"    : datetime.now(timezone.utc).isoformat()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict a single tweet / post text.

    Request body (JSON):
        { "text": "your tweet here", "model": "svm" }
        model options: "svm" | "lr" | "rf"  (default: best model)
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    raw       = str(data.get("text", "")).strip()
    model_key = str(data.get("model", BEST_MODEL)).lower()
    if not raw:
        return jsonify({"error": "Empty text provided"}), 400
    result = predict_single(raw, model_key)
    result["source"] = "api"
    return jsonify(result)


@app.route("/models", methods=["GET"])
def get_models():
    """Return all model metrics for dashboard charts."""
    output = {}
    for k, v in MODELS.items():
        output[k] = {
            "name"       : v["name"],
            "accuracy"   : v["accuracy"],
            "precision"  : v["precision"],
            "recall"     : v["recall"],
            "f1"         : v["f1"],
            "cv_score"   : v["cv_score"],
            "best_params": v["best_params"],
            "is_best"    : (k == BEST_MODEL)
        }
    return jsonify({"models": output, "best_model": BEST_MODEL})


@app.route("/live_feed", methods=["GET"])
def live_feed():
    """
    Returns real Reddit posts with ML predictions applied to each.

    Query params:
        ?refresh=true    — force fresh fetch from Reddit
        ?limit=20        — max results (default 20, max 50)
        ?model=svm       — which model to use for prediction
    """
    force     = request.args.get("refresh", "false").lower() == "true"
    limit     = min(int(request.args.get("limit", 20)), 50)
    model_key = request.args.get("model", BEST_MODEL)

    posts    = get_live_feed(force_refresh=force)
    enriched = []

    for p in posts[:limit]:
        pred = predict_single(p["title"], model_key)

        # Human-readable age
        ts  = p.get("created_utc", 0)
        age = ""
        if ts:
            diff = int(time.time() - ts)
            if diff < 3600:
                age = f"{diff // 60}m ago"
            elif diff < 86400:
                age = f"{diff // 3600}h ago"
            else:
                age = f"{diff // 86400}d ago"

        enriched.append({
            "id"               : p.get("id", ""),
            "title"            : p["title"],
            "platform"         : p.get("platform", "Reddit"),
            "subreddit"        : p.get("subreddit", ""),
            "score"            : p.get("score", 0),
            "comments"         : p.get("comments", 0),
            "upvote_ratio"     : p.get("upvote_ratio", 0),
            "url"              : p.get("url", ""),
            "age"              : age,
            "prediction"       : pred["prediction"],
            "confidence"       : pred["confidence"],
            "trending_prob"    : pred["trending_prob"],
            "non_trending_prob": pred["non_trending_prob"],
            "model_used"       : pred["model_used"]
        })

    return jsonify({
        "feed"         : enriched,
        "count"        : len(enriched),
        "model_used"   : MODELS[BEST_MODEL]["name"],
        "cache_age_sec": int(time.time() - cache_last_updated),
        "timestamp"    : datetime.now(timezone.utc).isoformat()
    })


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Predict multiple texts at once (max 50).

    Body: { "texts": ["text1", "text2", ...], "model": "svm" }
    """
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' list"}), 400
    texts     = data.get("texts", [])
    model_key = str(data.get("model", BEST_MODEL)).lower()
    out = []
    for raw in texts[:50]:
        r = predict_single(str(raw), model_key)
        out.append({
            "text"             : raw,
            "prediction"       : r["prediction"],
            "confidence"       : r["confidence"],
            "trending_prob"    : r["trending_prob"],
            "non_trending_prob": r["non_trending_prob"]
        })
    return jsonify({
        "results"   : out,
        "model_used": MODELS.get(model_key, MODELS[BEST_MODEL])["name"],
        "count"     : len(out)
    })


@app.route("/search_reddit", methods=["GET"])
def search_reddit():
    """
    Search Reddit for a keyword and return predictions on results.
    ?q=your+keyword&limit=10&model=svm
    """
    query     = request.args.get("q", "").strip()
    limit     = min(int(request.args.get("limit", 10)), 25)
    model_key = request.args.get("model", BEST_MODEL)

    if not query:
        return jsonify({"error": "Missing query param ?q="}), 400

    url = f"https://www.reddit.com/search.json?q={query}&sort=hot&limit={limit}"
    try:
        resp  = http_requests.get(url, headers=REDDIT_HEADERS, timeout=8)
        posts = resp.json().get("data", {}).get("children", [])
    except Exception as e:
        return jsonify({"error": f"Reddit fetch failed: {str(e)}"}), 500

    results = []
    for child in posts:
        p     = child.get("data", {})
        title = p.get("title", "").strip()
        if not title:
            continue
        pred = predict_single(title, model_key)
        results.append({
            "title"        : title,
            "subreddit"    : p.get("subreddit", ""),
            "score"        : p.get("score", 0),
            "comments"     : p.get("num_comments", 0),
            "url"          : "https://reddit.com" + p.get("permalink", ""),
            "prediction"   : pred["prediction"],
            "confidence"   : pred["confidence"],
            "trending_prob": pred["trending_prob"]
        })

    return jsonify({
        "query"     : query,
        "results"   : results,
        "count"     : len(results),
        "model_used": MODELS[BEST_MODEL]["name"]
    })


@app.route("/retrain", methods=["POST"])
def retrain():
    """
    Re-fetch fresh Reddit data and retrain all models in background.
    POST /retrain   (no body required)
    Returns immediately. Check /models after ~60s for new scores.
    """
    def _retrain():
        global MODELS, VECTORIZER, BEST_MODEL, LABEL_MAP
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        new_models, new_vec, new_best = train_models()
        MODELS     = new_models
        VECTORIZER = new_vec
        BEST_MODEL = new_best
        LABEL_MAP  = {1: "Trending", 0: "Non-Trending"}
        print("  [Retrain] Complete.")

    threading.Thread(target=_retrain, daemon=True).start()
    return jsonify({
        "status" : "retraining started",
        "message": "Models are being retrained in background using fresh Reddit data. "
                   "Check /models after ~60 seconds for updated accuracy scores."
    })


@app.route("/model_summary", methods=["GET"])
def model_summary():
    """Return a paper-ready model comparison table."""
    rows = []
    for k, v in MODELS.items():
        rows.append({
            "Model"    : v["name"],
            "Accuracy" : f"{v['accuracy']}%",
            "Precision": f"{v['precision']}%",
            "Recall"   : f"{v['recall']}%",
            "F1-Score" : f"{v['f1']}%",
            "CV Score" : f"{v['cv_score']}%",
            "Best"     : "Yes" if k == BEST_MODEL else "No"
        })
    return jsonify({"summary": rows})


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "─"*62)
    print("  API ENDPOINTS:")
    print("  GET  /health                   — Server status")
    print("  POST /predict                  — Predict one post/tweet")
    print("  POST /batch_predict            — Predict many texts")
    print("  GET  /live_feed                — Real-time Reddit + predictions")
    print("  GET  /live_feed?refresh=true   — Force fresh Reddit fetch")
    print("  GET  /search_reddit?q=AI       — Search Reddit + predict")
    print("  GET  /models                   — All model accuracy metrics")
    print("  GET  /model_summary            — Paper-ready table")
    print("  POST /retrain                  — Retrain with fresh data")
    print("─"*62 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
