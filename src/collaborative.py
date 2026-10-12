"""
Item-Based Collaborative Filtering and Hybrid Recommendation Engine.
Leverages user-item interaction matrix from MovieLens ratings to compute
collaborative similarity and fuses it with content-based signals.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class CollaborativeRecommender:
    """
    Item-based Collaborative Filtering engine utilizing user rating patterns.
    """
    def __init__(self, ratings_df: pd.DataFrame, min_ratings_per_movie: int = 5):
        self.ratings_df = ratings_df.copy()
        self.min_ratings_per_movie = min_ratings_per_movie
        self.item_user_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.similarity_matrix = None
        self._build_model()

    def _build_model(self):
        """Constructs the user-item interaction matrix and computes item similarity."""
        # Filter out cold-start movies with fewer than min_ratings_per_movie
        movie_counts = self.ratings_df['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        filtered_ratings = self.ratings_df[self.ratings_df['movieId'].isin(valid_movies)]

        # Pivot to create Item-User matrix (rows: movieId, cols: userId)
        pivot = filtered_ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
        
        self.movie_ids = pivot.index.values
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {i: mid for i, mid in enumerate(self.movie_ids)}
        
        # Compute item-item cosine similarity
        sparse_matrix = csr_matrix(pivot.values)
        self.similarity_matrix = cosine_similarity(sparse_matrix, dense_output=True)

    def get_similar_movies(self, movie_id: int, top_n: int = 10):
        """Returns list of (movie_id, similarity_score) for a given movie."""
        if movie_id not in self.movie_id_to_idx:
            return []
        
        idx = self.movie_id_to_idx[movie_id]
        scores = self.similarity_matrix[idx].copy()
        scores[idx] = -1.0  # Exclude self
        
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.idx_to_movie_id[i], float(scores[i])) for i in top_indices if scores[i] > 0]


class HybridRecommender:
    """
    Hybrid recommendation engine combining Content-Based TF-IDF and
    Collaborative Item-Item filtering signals with configurable blending weights.
    """
    def __init__(self, content_recommender, collaborative_recommender, movies_df: pd.DataFrame):
        self.content = content_recommender
        self.collab = collaborative_recommender
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_id_to_row = {mid: i for i, mid in enumerate(self.movies_df['movieId'])}

    def get_hybrid_recommendations(
        self,
        movie_title_or_id,
        top_n: int = 6,
        alpha: float = 0.65,
        min_rating: float = 0.0,
        min_votes: int = 0,
        year_range: tuple = None,
        genre_filter: str = None
    ):
        """
        Computes hybrid recommendations blending content and collaborative scores.

        Score = alpha * Content_Score + (1 - alpha) * Collaborative_Score
        """
        # Resolve target movie index
        if isinstance(movie_title_or_id, str):
            target_idx = self.content.find_movie_by_title(movie_title_or_id)
            if target_idx is None:
                raise ValueError(f"Movie '{movie_title_or_id}' not found.")
        else:
            target_idx = int(movie_title_or_id)

        target_movie = self.movies_df.iloc[target_idx]
        target_movie_id = target_movie['movieId']

        # 1. Content-based similarity scores for all items
        target_vec = self.content.tfidf_matrix[target_idx]
        content_sims = cosine_similarity(target_vec, self.content.tfidf_matrix).flatten()

        # 2. Collaborative similarity scores
        collab_sims = np.zeros(len(self.movies_df))
        if target_movie_id in self.collab.movie_id_to_idx:
            collab_idx = self.collab.movie_id_to_idx[target_movie_id]
            collab_scores = self.collab.similarity_matrix[collab_idx]
            
            for c_i, c_score in enumerate(collab_scores):
                mid = self.collab.idx_to_movie_id[c_i]
                if mid in self.movie_id_to_row:
                    row_i = self.movie_id_to_row[mid]
                    collab_sims[row_i] = max(0.0, float(c_score))

        # 3. Blend scores
        has_collab = target_movie_id in self.collab.movie_id_to_idx
        if has_collab:
            hybrid_scores = (alpha * content_sims) + ((1.0 - alpha) * collab_sims)
        else:
            hybrid_scores = content_sims

        hybrid_scores[target_idx] = -1.0  # Exclude self

        # 4. Filter criteria
        eligible_mask = np.ones(len(self.movies_df), dtype=bool)
        eligible_mask[target_idx] = False

        if min_rating > 0.0:
            eligible_mask &= (self.movies_df['avg_rating'].values >= min_rating)
        if min_votes > 0:
            eligible_mask &= (self.movies_df['vote_count'].values >= min_votes)
        if year_range is not None:
            min_y, max_y = year_range
            eligible_mask &= (self.movies_df['year'].values >= min_y) & (self.movies_df['year'].values <= max_y)
        if genre_filter:
            eligible_mask &= self.movies_df['genre_list'].apply(lambda lst: genre_filter in lst).values

        filtered_indices = np.where(eligible_mask)[0]
        if len(filtered_indices) == 0:
            return pd.DataFrame()

        sub_scores = hybrid_scores[filtered_indices]
        top_k_sub = np.argsort(sub_scores)[::-1][:top_n]
        best_indices = filtered_indices[top_k_sub]

        results = self.movies_df.iloc[best_indices].copy()
        results['similarity_score'] = np.round(hybrid_scores[best_indices], 3)
        results['content_score'] = np.round(content_sims[best_indices], 3)
        results['collab_score'] = np.round(collab_sims[best_indices], 3)
        results['algorithm'] = 'Hybrid (Content + Collaborative)' if has_collab else 'Content-Based Fallback'

        return results
