import os
import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.recommender import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender, HybridRecommender
from src.evaluate import precision_at_k, recall_at_k, reciprocal_rank


@pytest.fixture(scope="module")
def setup_models():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ml-latest-small')
    movies_df = load_and_preprocess_data(data_dir)
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)

    content = ContentBasedRecommender(movies_df, max_features=3000)
    collab = CollaborativeRecommender(ratings_df, min_ratings_per_movie=5)
    hybrid = HybridRecommender(content, collab, movies_df)
    return content, collab, hybrid, movies_df


def test_collaborative_recommender(setup_models):
    """Verifies item-item collaborative similarity produces valid scores."""
    _, collab, _, _ = setup_models
    assert collab.similarity_matrix is not None
    assert len(collab.movie_ids) > 1000

    # Test Toy Story (movieId: 1)
    similar = collab.get_similar_movies(1, top_n=5)
    assert len(similar) == 5
    for mid, score in similar:
        assert isinstance(mid, (int, np.integer))
        assert 0.0 <= score <= 1.0


def test_hybrid_recommender(setup_models):
    """Verifies hybrid recommender fuses content and collaborative scores."""
    _, _, hybrid, movies_df = setup_models
    recs = hybrid.get_hybrid_recommendations("Toy Story (1995)", top_n=5, alpha=0.6)

    assert isinstance(recs, pd.DataFrame)
    assert len(recs) == 5
    assert 'similarity_score' in recs.columns
    assert 'content_score' in recs.columns
    assert 'collab_score' in recs.columns

    # Verify descending score order
    scores = recs['similarity_score'].tolist()
    assert scores == sorted(scores, reverse=True)


def test_evaluation_metrics():
    """Unit tests for offline IR evaluation metrics."""
    recommended = [10, 20, 30, 40, 50]
    relevant = {20, 40, 99}

    # Precision@5: 2 hits out of 5 = 0.4
    p5 = precision_at_k(recommended, relevant, k=5)
    assert p5 == pytest.approx(0.4)

    # Precision@2: 1 hit (item 20) out of 2 = 0.5
    p2 = precision_at_k(recommended, relevant, k=2)
    assert p2 == pytest.approx(0.5)

    # Recall@5: 2 hits out of 3 relevant = 2/3
    r5 = recall_at_k(recommended, relevant, k=5)
    assert r5 == pytest.approx(2 / 3)

    # MRR: first hit is at rank 2 (item 20), so 1/2 = 0.5
    mrr = reciprocal_rank(recommended, relevant, k=5)
    assert mrr == pytest.approx(0.5)
