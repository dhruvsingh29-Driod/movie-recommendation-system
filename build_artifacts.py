"""
Builds and serializes recommendation artifacts for high-speed retrieval in Streamlit.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.data_loader import load_and_preprocess_data
from src.recommender import ContentBasedRecommender

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts')

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_all_artifacts():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()
    print("=" * 60)
    print("[*] MOVIELENS 100K RECOMMENDATION ENGINE BUILDER")
    print("=" * 60)

    # 1. Load and Preprocess Data
    print("\n[1/4] Ingesting and featurizing MovieLens 100K data...")
    df = load_and_preprocess_data()
    print(f"      ✓ Loaded {len(df):,} movies, {df['rating_count'].sum():,} total ratings.")

    # 2. Build Recommender Model
    print("\n[2/4] Training TF-IDF vectorizer and generating feature matrix...")
    recommender = ContentBasedRecommender(df, max_features=8000)
    print(f"      ✓ Feature matrix shape: {recommender.tfidf_matrix.shape}")

    # 3. Precompute Top 50 Similarities
    print("\n[3/4] Pre-computing Top-50 nearest neighbors (cosine similarity)...")
    top50_map = {}
    batch_size = 1000
    n_movies = len(df)
    
    # Process in batches for fast vectorization & memory safety
    for start_idx in range(0, n_movies, batch_size):
        end_idx = min(start_idx + batch_size, n_movies)
        sub_matrix = recommender.tfidf_matrix[start_idx:end_idx]
        sim_batch = cosine_similarity(sub_matrix, recommender.tfidf_matrix)

        for row_offset in range(len(sim_batch)):
            global_idx = start_idx + row_offset
            scores = sim_batch[row_offset]
            # Exclude self
            scores[global_idx] = -1.0
            # Get top 50
            top_indices = scores.argsort()[::-1][:50]
            top_scores = scores[top_indices]
            
            top50_map[global_idx] = [
                (int(idx), float(score)) for idx, score in zip(top_indices, top_scores) if score > 0.0
            ]

    print(f"      ✓ Successfully indexed {len(top50_map):,} movie similarity vectors.")

    # 4. Generate EDA & Dataset Analytics
    print("\n[4/4] Computing exploratory data analysis (EDA) metrics...")
    all_genres = []
    for g_list in df['genre_list']:
        all_genres.extend(g_list)
    genre_series = pd.Series(all_genres).value_counts()

    eda_stats = {
        'total_movies': int(len(df)),
        'total_ratings': int(df['rating_count'].sum()),
        'avg_rating_overall': round(float(df[df['avg_rating'] > 0]['avg_rating'].mean()), 2),
        'top_genres': genre_series.to_dict(),
        'years_min': int(df['year'].dropna().min()),
        'years_max': int(df['year'].dropna().max()),
        'rating_distribution': df['avg_rating'].value_counts().sort_index().to_dict()
    }

    # Save artifacts
    movies_path = os.path.join(ARTIFACTS_DIR, 'movies_df.pkl')
    recommender_path = os.path.join(ARTIFACTS_DIR, 'recommender.pkl')
    top50_path = os.path.join(ARTIFACTS_DIR, 'top50_similarities.pkl')
    stats_path = os.path.join(ARTIFACTS_DIR, 'eda_stats.json')

    with open(movies_path, 'wb') as f:
        pickle.dump(df, f)

    with open(recommender_path, 'wb') as f:
        pickle.dump(recommender, f)

    with open(top50_path, 'wb') as f:
        pickle.dump(top50_map, f)

    with open(stats_path, 'w') as f:
        json.dump(eda_stats, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[*] Artifacts generated in {elapsed:.2f} seconds!")
    print(f"    Output directory: {ARTIFACTS_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    build_all_artifacts()
