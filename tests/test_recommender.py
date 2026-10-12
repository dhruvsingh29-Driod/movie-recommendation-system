import os
import pytest
import pandas as pd
from src.data_loader import load_and_preprocess_data
from src.recommender import ContentBasedRecommender


@pytest.fixture(scope="module")
def dataset():
    """Loads and returns the preprocessed MovieLens dataset."""
    df = load_and_preprocess_data()
    return df


@pytest.fixture(scope="module")
def recommender(dataset):
    """Initializes and returns the fitted ContentBasedRecommender."""
    rec = ContentBasedRecommender(dataset, max_features=4000)
    return rec


def test_dataset_integrity(dataset):
    """Verifies dataset has expected columns and non-empty rows."""
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 9000
    expected_cols = {'movieId', 'title', 'genres', 'genre_list', 'soup', 'avg_rating', 'rating_count'}
    assert expected_cols.issubset(set(dataset.columns))
    assert dataset['soup'].isna().sum() == 0


def test_movie_lookup(recommender):
    """Tests flexible title search (exact, clean, substring)."""
    # Exact lookup
    idx1 = recommender.find_movie_by_title("Toy Story (1995)")
    assert idx1 is not None

    # Title without year
    idx2 = recommender.find_movie_by_title("Toy Story")
    assert idx2 is not None

    # Case insensitive
    idx3 = recommender.find_movie_by_title("toy story")
    assert idx3 == idx1

    # Non-existent movie
    assert recommender.find_movie_by_title("NonExistentMovie999") is None


def test_recommendation_bounds_and_ranking(recommender):
    """Ensures cosine similarity scores are bounded between 0 and 1, and sorted descending."""
    target, recs = recommender.get_recommendations("Toy Story (1995)", top_n=6)
    
    assert len(recs) == 6
    assert recs[0]['movieId'] != target['movieId']

    prev_score = 1.0
    for r in recs:
        score = r['similarity_score']
        assert 0.0 <= score <= 1.0
        assert score <= prev_score  # monotonic descending
        prev_score = score
        assert 'title' in r
        assert 'match_percentage' in r
        assert 'explanation' in r


def test_genre_filter(recommender):
    """Verifies that genre filtering strictly returns movies matching the requested genre."""
    target, recs = recommender.get_recommendations(
        "Toy Story (1995)",
        top_n=5,
        genre_filter="Animation"
    )
    for r in recs:
        assert "Animation" in r['genre_list']


def test_min_rating_filter(recommender):
    """Verifies that rating threshold filter is strictly enforced."""
    target, recs = recommender.get_recommendations(
        "Toy Story (1995)",
        top_n=5,
        min_rating=3.5
    )
    for r in recs:
        assert r['avg_rating'] >= 3.5


def test_invalid_query_raises_error(recommender):
    """Checks that invalid queries raise appropriate ValueError."""
    with pytest.raises(ValueError):
        recommender.get_recommendations("NonExistentMovieXYZ", top_n=5)
