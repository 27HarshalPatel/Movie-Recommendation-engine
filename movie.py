from flask import Flask, request, render_template_string
import math
from collections import defaultdict

"""
Movie Recommendation Engine – Flask Web App (Prompt-driven Rating)
------------------------------------------------------------------
Features:
- User enters a movie name (free text) → system matches dataset.
- User provides a rating (1–5) for that movie.
- User enters a natural language recommendation prompt.
- Hybrid ML recommendations are generated and boosted by the prompt.
"""

# -------------------------- Sample Dataset --------------------------
MOVIES = [
    {"id": 1, "title": "Inception", "year": 2010, "genres": ["Action", "Sci-Fi", "Thriller"], "tags": ["dream", "heist", "mind-bending", "nolan"]},
    {"id": 2, "title": "The Dark Knight", "year": 2008, "genres": ["Action", "Crime", "Drama"], "tags": ["batman", "joker", "nolan"]},
    {"id": 3, "title": "Interstellar", "year": 2014, "genres": ["Adventure", "Drama", "Sci-Fi"], "tags": ["space", "time", "nolan", "wormhole"]},
    {"id": 4, "title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"], "tags": ["simulation", "kung fu", "ai"]},
    {"id": 5, "title": "Parasite", "year": 2019, "genres": ["Drama", "Thriller"], "tags": ["class", "korean", "academy awards"]},
]

COMMUNITY = [
    {"userId": "u_alex", "ratings": {1: 5, 2: 5, 3: 5, 4: 5}},
    {"userId": "u_bianca", "ratings": {5: 5, 3: 4}},
]

# -------------------------- Utils --------------------------
def cosine_sim(a, b):
    dot, na, nb = 0, 0, 0
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va, vb = a.get(k, 0), b.get(k, 0)
        dot += va * vb
        na += va * va
        nb += vb * vb
    if na == 0 or nb == 0:
        return 0
    return dot / (math.sqrt(na) * math.sqrt(nb))

# -------------------------- TF-IDF --------------------------
def build_tfidf(movies):
    df = defaultdict(int)
    docs = []
    for m in movies:
        tokens = [t.lower() for t in (m["genres"] + m["tags"])]
        docs.append((m["id"], tokens))
        for t in set(tokens):
            df[t] += 1
    N = len(movies)
    idf = {t: math.log((N + 1) / (dfi + 1)) + 1 for t, dfi in df.items()}
    vectors = {}
    for mid, tokens in docs:
        counts = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        vec = {t: c * idf.get(t, 0) for t, c in counts.items()}
        vectors[mid] = vec
    return vectors

def build_user_profile(tfidf_vectors, ratings):
    liked = [mid for mid, r in ratings.items() if r >= 4]
    if not liked:
        return {}
    acc = defaultdict(float)
    for mid in liked:
        vec = tfidf_vectors.get(mid, {})
        for t, w in vec.items():
            acc[t] += w
    norm = math.sqrt(sum(v * v for v in acc.values())) or 1
    return {t: v / norm for t, v in acc.items()}

# -------------------------- CF & CB --------------------------
def predict_cf(ratings, community, k=3):
    sims = []
    for u in community:
        sim = cosine_sim(ratings, u["ratings"])
        sims.append((u["userId"], sim, u["ratings"]))
    sims.sort(key=lambda x: x[1], reverse=True)
    neighbors = [s for s in sims if s[1] > 0][:k]
    pred = {}
    for m in [mv["id"] for mv in MOVIES]:
        if m in ratings:
            continue
        num, den = 0, 0
        for uid, sim, r in neighbors:
            if m in r:
                num += sim * r[m]
                den += abs(sim)
        if den > 0:
            pred[m] = num / den
    return pred

def predict_cb(tfidf_vectors, user_profile, ratings):
    scores = {}
    for m in MOVIES:
        mid = m["id"]
        if mid in ratings:
            continue
        sim = cosine_sim(user_profile, tfidf_vectors.get(mid, {}))
        scores[mid] = sim * 5
    return scores

def hybrid_scores(cf_scores, cb_scores, alpha=0.6):
    results = {}
    for m in [mv["id"] for mv in MOVIES]:
        cf = cf_scores.get(m, 0)
        cb = cb_scores.get(m, 0)
        if cf or cb:
            results[m] = alpha * cf + (1 - alpha) * cb
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# -------------------------- Prompt Boost --------------------------
def apply_prompt_boost(hybrid, prompt):
    if not prompt:
        return hybrid
    q = prompt.lower()
    boosted = {}
    for mid, score in hybrid.items():
        m = next(m for m in MOVIES if m["id"] == mid)
        hay = " ".join([m["title"]] + m["genres"] + m["tags"]).lower()
        boost = 2.0 if q in hay else 0
        boosted[mid] = score + boost
    return dict(sorted(boosted.items(), key=lambda x: x[1], reverse=True))

# -------------------------- Flask App --------------------------
app = Flask(__name__)

template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Movie Recommendation Engine</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
  <div class="container py-5">
    <h1 class="mb-4">🎬 Movie Recommendation Engine</h1>
    <form method="POST" class="card p-4 shadow-sm mb-4">
      <div class="mb-3">
        <label class="form-label">Which movie would you like to rate?</label>
        <input type="text" name="movie_name" class="form-control" placeholder="e.g. Inception" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Your rating (1-5)</label>
        <input type="number" name="rating" min="1" max="5" class="form-control" required>
      </div>
      <div class="mb-3">
        <label class="form-label">Your recommendation prompt</label>
        <input type="text" name="prompt" class="form-control" placeholder="e.g. recommend me a sci-fi movie with AI">
      </div>
      <button type="submit" class="btn btn-primary">Get Recommendations</button>
    </form>

    {% if error %}
      <div class="alert alert-danger">{{error}}</div>
    {% endif %}

    {% if recs %}
      <h2>Recommended for You</h2>
      <div class="row">
        {% for movie, score, boosted in recs %}
          <div class="col-md-6">
            <div class="card mb-3 shadow-sm">
              <div class="card-body">
                <h5 class="card-title">{{movie['title']}} ({{movie['year']}})</h5>
                <h6 class="card-subtitle mb-2 text-muted">Score: {{"%.2f" % score}}</h6>
                <p class="card-text">Genres: {{", ".join(movie['genres'])}}</p>
                <p class="card-text"><small>Tags: {{", ".join(movie['tags'])}}</small></p>
                {% if boosted %}<span class="badge bg-purple">🔮 Boosted by Prompt</span>{% endif %}
              </div>
            </div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    recs, error = None, None
    if request.method == "POST":
        movie_name = request.form.get("movie_name", "").strip().lower()
        rating = int(request.form.get("rating"))
        prompt = request.form.get("prompt", "")

        # Match movie
        movie = next((m for m in MOVIES if movie_name in m["title"].lower()), None)
        if not movie:
            error = f"Movie '{movie_name}' not found in dataset. Try Inception, Interstellar, etc."
        else:
            ratings = {movie["id"]: rating}
            tfidf = build_tfidf(MOVIES)
            user_profile = build_user_profile(tfidf, ratings)
            cf = predict_cf(ratings, COMMUNITY)
            cb = predict_cb(tfidf, user_profile, ratings)
            hybrid = hybrid_scores(cf, cb, alpha=0.6)
            boosted = apply_prompt_boost(hybrid, prompt)

            recs = []
            for mid, score in list(boosted.items())[:5]:
                m = next(m for m in MOVIES if m["id"] == mid)
                boosted_flag = prompt.lower() in " ".join([m["title"]] + m["genres"] + m["tags"]).lower()
                recs.append((m, score, boosted_flag))
    return render_template_string(template, recs=recs, movies=MOVIES, error=error)

if __name__ == "__main__":
    app.run(debug=True)
