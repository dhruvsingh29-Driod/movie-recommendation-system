"""
Content-based recommendation engine utilizing TF-IDF and Cosine Similarity.
Provides top-N personalized movie recommendations with explainability and filtering.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame, max_features: int = 8000):
        self.df = df.copy().reset_index(drop=True)
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=self.max_features, ngram_range=(1, 2))
        self.tfidf_matrix = None
        self._build_model()

    def _build_model(self):
        """Fits TF-IDF vectorizer on the metadata soup."""
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['soup'])

    def find_movie_by_title(self, query: str):
        """
        Performs flexible matching for movie titles.
        Returns matching row index or None.
        """
        if not query:
            return None
            
        clean_q = query.strip().lower()
        
        # 1. Exact match on full title
        exact = self.df[self.df['title'].str.lower() == clean_q]
        if not exact.empty:
            return exact.index[0]
            
        # 2. Exact match on clean_title (without year)
        exact_clean = self.df[self.df['clean_title'].str.lower() == clean_q]
        if not exact_clean.empty:
            return exact_clean.index[0]

        # 3. Starts-with match
        starts = self.df[self.df['clean_title'].str.lower().str.startswith(clean_q)]
        if not starts.empty:
            return starts.index[0]

        # 4. Substring contains match
        contains = self.df[self.df['title'].str.lower().str.contains(clean_q, regex=False)]
        if not contains.empty:
            return contains.index[0]

        return None

    def get_recommendations(
        self,
        movie_identifier,
        top_n: int = 6,
        min_rating: float = 0.0,
        min_votes: int = 0,
        year_range: tuple = None,
        genre_filter: str = None,
        use_rating_weight: bool = False,
        weight_alpha: float = 0.85
    ):
        """
        Computes top-N recommendations for a target movie using cosine similarity.
        
        Parameters:
            movie_identifier: Title string or integer index
            top_n: Number of recommendations to return
            min_rating: Minimum average user rating filter (0.0 to 5.0)
            min_votes: Minimum number of ratings received
            year_range: Tuple of (min_year, max_year)
            genre_filter: Only return movies matching this specific genre
            use_rating_weight: If True, combines cosine similarity with normalized rating
            weight_alpha: Weight given to similarity vs rating (0.0 to 1.0)
        """
        if isinstance(movie_identifier, str):
            idx = self.find_movie_by_title(movie_identifier)
            if idx is None:
                raise ValueError(f"Movie '{movie_identifier}' not found in catalog.")
        elif isinstance(movie_identifier, (int, np.integer)):
            idx = int(movie_identifier)
        else:
            raise TypeError("Movie identifier must be title string or index integer.")

        target_movie = self.df.iloc[idx]
        target_vec = self.tfidf_matrix[idx]

        # Calculate cosine similarity of target movie against all movies
        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        # Combine with rating if requested
        if use_rating_weight:
            normalized_ratings = self.df['avg_rating'].values / 5.0
            composite_scores = (weight_alpha * sim_scores) + ((1 - weight_alpha) * normalized_ratings)
            effective_scores = composite_scores
        else:
            effective_scores = sim_scores

        # Rank candidates (descending)
        ranked_indices = effective_scores.argsort()[::-1]

        recommendations = []
        for i in ranked_indices:
            # Exclude target movie itself
            if i == idx:
                continue

            candidate = self.df.iloc[i]

            # Apply user filters
            if candidate['avg_rating'] < min_rating:
                continue
            if candidate['rating_count'] < min_votes:
                continue
            if year_range and pd.notna(candidate['year']):
                if candidate['year'] < year_range[0] or candidate['year'] > year_range[1]:
                    continue
            if genre_filter and genre_filter != 'All Genres':
                if genre_filter not in candidate['genre_list']:
                    continue

            # Explainability: Intersecting attributes
            shared_genres = sorted(list(set(target_movie['genre_list']) & set(candidate['genre_list'])))
            shared_tags = sorted(list(set(target_movie['tag_list']) & set(candidate['tag_list'])))

            explanation_parts = []
            if shared_genres:
                explanation_parts.append(f"Genres: {', '.join(shared_genres)}")
            if shared_tags:
                explanation_parts.append(f"Themes: {', '.join(shared_tags)}")
            
            explanation_text = " | ".join(explanation_parts) if explanation_parts else "High thematic similarity"

            recommendations.append({
                'movieId': int(candidate['movieId']),
                'title': candidate['title'],
                'clean_title': candidate['clean_title'],
                'year': candidate['year'],
                'genres': candidate['genres'],
                'genre_list': candidate['genre_list'],
                'tag_list': candidate['tag_list'],
                'avg_rating': float(candidate['avg_rating']),
                'rating_count': int(candidate['rating_count']),
                'similarity_score': float(sim_scores[i]),
                'match_percentage': round(float(sim_scores[i]) * 100, 1),
                'shared_genres': shared_genres,
                'shared_tags': shared_tags,
                'explanation': explanation_text,
                'imdbId': candidate.get('imdbId'),
                'tmdbId': candidate.get('tmdbId')
            })

            if len(recommendations) >= top_n:
                break

        return target_movie, recommendations
