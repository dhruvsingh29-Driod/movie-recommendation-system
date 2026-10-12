"""
Data loading and feature engineering module for MovieLens 100K dataset.
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'ml-latest-small')

def load_and_preprocess_data(data_dir=DATA_DIR):
    """
    Loads movies.csv, tags.csv, ratings.csv, and links.csv,
    performs data cleaning, aggregates user tags and ratings,
    and constructs the feature 'soup' for content-based vectorization.
    """
    movies_path = os.path.join(data_dir, 'movies.csv')
    tags_path = os.path.join(data_dir, 'tags.csv')
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    links_path = os.path.join(data_dir, 'links.csv')

    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"MovieLens data not found in {data_dir}. Run download first.")

    movies_df = pd.read_csv(movies_path)
    tags_df = pd.read_csv(tags_path) if os.path.exists(tags_path) else pd.DataFrame(columns=['movieId', 'tag'])
    ratings_df = pd.read_csv(ratings_path) if os.path.exists(ratings_path) else pd.DataFrame(columns=['movieId', 'rating'])
    links_df = pd.read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame(columns=['movieId', 'imdbId', 'tmdbId'])

    # 1. Process Movie Titles and Extract Release Year
    # Format typically: "Toy Story (1995)" or "Seven (a.k.a. Se7en) (1995)"
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    # 2. Process Genres
    # Convert 'Adventure|Animation|Children' -> list and space-separated lowercase string
    movies_df['genre_list'] = movies_df['genres'].apply(
        lambda g: [] if g == '(no genres listed)' else [x.strip() for x in str(g).split('|')]
    )
    movies_df['genres_tokens'] = movies_df['genre_list'].apply(
        lambda lst: ' '.join([g.lower().replace('-', '').replace(' ', '') for g in lst])
    )

    # 3. Aggregate User Tags per Movie
    if not tags_df.empty:
        tags_df['clean_tag'] = tags_df['tag'].fillna('').astype(str).str.lower().str.strip()
        tags_df['clean_tag'] = tags_df['clean_tag'].apply(lambda t: re.sub(r'[^a-zA-Z0-9\s]', '', t))
        
        tags_grouped = tags_df.groupby('movieId').agg(
            tag_string=('clean_tag', lambda x: ' '.join(set([t for t in x if t]))),
            tag_list=('clean_tag', lambda x: sorted(list(set([t for t in x if t]))))
        ).reset_index()
    else:
        tags_grouped = pd.DataFrame(columns=['movieId', 'tag_string', 'tag_list'])

    # 4. Aggregate Ratings (Mean rating and Vote count)
    if not ratings_df.empty:
        ratings_grouped = ratings_df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            rating_count=('rating', 'count')
        ).reset_index()
        ratings_grouped['avg_rating'] = ratings_grouped['avg_rating'].round(1)
    else:
        ratings_grouped = pd.DataFrame(columns=['movieId', 'avg_rating', 'rating_count'])

    # 5. Merge everything
    df = movies_df.merge(tags_grouped, on='movieId', how='left')
    df = df.merge(ratings_grouped, on='movieId', how='left')
    if not links_df.empty:
        df = df.merge(links_df[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
    else:
        df['imdbId'] = None
        df['tmdbId'] = None

    # Handle missing values cleanly
    df['tag_string'] = df['tag_string'].fillna('')
    df['tag_list'] = df['tag_list'].apply(lambda x: x if isinstance(x, list) else [])
    df['avg_rating'] = df['avg_rating'].fillna(0.0)
    df['rating_count'] = df['rating_count'].fillna(0).astype(int)

    # 6. Feature Engineering: Content Soup
    # We weight genres by repeating them to give genre affinity strong anchor,
    # enriched with distinctive user-provided metadata tags.
    df['soup'] = (
        df['genres_tokens'] + ' ' + 
        df['genres_tokens'] + ' ' + 
        df['tag_string']
    ).str.strip()

    # Fallback for movies with empty soup
    df['soup'] = df['soup'].apply(lambda s: s if s else 'movie')

    return df

if __name__ == '__main__':
    df = load_and_preprocess_data()
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(df[['movieId', 'title', 'year', 'avg_rating', 'rating_count', 'soup']].head(5))
