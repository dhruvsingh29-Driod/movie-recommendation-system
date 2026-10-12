"""
Production Recommender REST API Microservice.
Provides headless JSON endpoints for model inference, search, and evaluation.
Runs using Python standard library http.server for zero external runtime dependencies.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd

from src.data_loader import load_and_preprocess_data
from src.recommender import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender, HybridRecommender
from src.evaluate import evaluate_recommenders

# Preload data & models
print("Initializing Recommender API models...")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ml-latest-small')
MOVIES_DF = load_and_preprocess_data(DATA_DIR)
CONTENT_ENGINE = ContentBasedRecommender(MOVIES_DF)

ratings_path = os.path.join(DATA_DIR, 'ratings.csv')
ratings_df = pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame()
COLLAB_ENGINE = CollaborativeRecommender(ratings_df) if not ratings_df.empty else None
HYBRID_ENGINE = HybridRecommender(CONTENT_ENGINE, COLLAB_ENGINE, MOVIES_DF) if COLLAB_ENGINE else None

print(f"Models initialized. Catalog size: {len(MOVIES_DF)} movies.")


class RecommenderAPIHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    def _send_html(self, status_code, html_str):
        self.send_response(status_code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html_str.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # 1. Health check
        if path == '/health' or path == '/api/v1/health':
            self._send_json(200, {
                "status": "healthy",
                "service": "Movie Recommendation API",
                "catalog_size": len(MOVIES_DF),
                "version": "1.2.0"
            })
            return

        # 2. Movie search / catalog
        elif path == '/api/v1/movies':
            q = params.get('q', [''])[0].strip().lower()
            limit = int(params.get('limit', ['10'])[0])
            if q:
                matches = MOVIES_DF[MOVIES_DF['title'].str.lower().str.contains(q, regex=False)].head(limit)
            else:
                matches = MOVIES_DF.head(limit)
            
            records = matches[['movieId', 'title', 'year', 'genres', 'avg_rating', 'vote_count']].to_dict(orient='records')
            self._send_json(200, {"count": len(records), "movies": records})
            return

        # 3. Recommendations endpoint
        elif path == '/api/v1/recommend':
            title_query = params.get('movie', [''])[0].strip()
            top_n = int(params.get('top_n', ['6'])[0])
            mode = params.get('mode', ['hybrid' if HYBRID_ENGINE else 'content'])[0].lower()

            if not title_query:
                self._send_json(400, {"error": "Missing 'movie' query parameter. Example: ?movie=Toy%20Story&top_n=5"})
                return

            try:
                if mode == 'hybrid' and HYBRID_ENGINE:
                    recs = HYBRID_ENGINE.get_hybrid_recommendations(title_query, top_n=top_n)
                else:
                    recs = CONTENT_ENGINE.get_recommendations(title_query, top_n=top_n)

                if recs.empty:
                    self._send_json(404, {"error": f"No recommendations found for '{title_query}'"})
                    return

                items = recs[['movieId', 'title', 'year', 'genres', 'avg_rating', 'vote_count', 'similarity_score']].to_dict(orient='records')
                self._send_json(200, {
                    "query": title_query,
                    "algorithm": mode,
                    "count": len(items),
                    "recommendations": items
                })
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        # 4. Evaluation Benchmark Endpoint
        elif path == '/api/v1/evaluate':
            k = int(params.get('k', ['5'])[0])
            users = int(params.get('users', ['25'])[0])
            benchmark = evaluate_recommenders(DATA_DIR, k=k, test_users=users)
            self._send_json(200, benchmark)
            return

        # 5. Interactive Docs
        elif path == '/docs' or path == '/':
            doc_html = """<!DOCTYPE html>
<html>
<head>
  <title>Movie Recommendation Engine | REST API Docs</title>
  <meta charset="utf-8" />
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; padding: 2rem; max-width: 900px; margin: auto; }
    h1 { color: #818cf8; border-bottom: 1px solid #1e293b; padding-bottom: 0.75rem; }
    .badge { background: #4f46e5; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .route { background: #131d31; border: 1px solid #1e293b; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }
    code { background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }
    a { color: #60a5fa; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>🎬 Movie Recommender REST API <span class="badge">v1.2.0</span></h1>
  <p>Production inference and evaluation microservice built on MovieLens 100K.</p>

  <div class="route">
    <h3><span class="badge">GET</span> <code>/health</code></h3>
    <p>Service health check and active catalog count.</p>
    <p>Try: <a href="/health" target="_blank">/health</a></p>
  </div>

  <div class="route">
    <h3><span class="badge">GET</span> <code>/api/v1/recommend?movie={title}&top_n={n}&mode={hybrid|content}</code></h3>
    <p>Get personalized recommendations with similarity ranking.</p>
    <p>Try: <a href="/api/v1/recommend?movie=Toy%20Story&top_n=5" target="_blank">/api/v1/recommend?movie=Toy%20Story&top_n=5</a></p>
  </div>

  <div class="route">
    <h3><span class="badge">GET</span> <code>/api/v1/movies?q={query}&limit={n}</code></h3>
    <p>Search movie metadata catalog.</p>
    <p>Try: <a href="/api/v1/movies?q=matrix" target="_blank">/api/v1/movies?q=matrix</a></p>
  </div>

  <div class="route">
    <h3><span class="badge">GET</span> <code>/api/v1/evaluate?k=5</code></h3>
    <p>Run offline Precision@K, Recall@K, and MRR evaluation benchmarks.</p>
    <p>Try: <a href="/api/v1/evaluate?k=5&users=10" target="_blank">/api/v1/evaluate?k=5&users=10</a></p>
  </div>
</body>
</html>"""
            self._send_html(200, doc_html)
            return

        else:
            self._send_json(404, {"error": "Endpoint not found. Visit /docs for API documentation."})


def run_api_server(port=8000):
    server = HTTPServer(('127.0.0.1', port), RecommenderAPIHandler)
    print(f"🎬 Movie Recommender REST API running at http://127.0.0.1:{port}")
    print(f"   API Documentation available at http://127.0.0.1:{port}/docs")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
        server.server_close()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_api_server(port)
