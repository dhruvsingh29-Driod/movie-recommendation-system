"""
Poster and metadata enrichment service.
Supports TMDB API poster fetching with an elegant, responsive fallback generator
and direct trailer / IMDb deep links.
"""

import os
import urllib.parse
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
POSTER_CACHE = {}

def get_imdb_url(imdb_id):
    """Returns official IMDb URL."""
    if not imdb_id or str(imdb_id).lower() == 'nan':
        return None
    try:
        clean_id = f"tt{int(imdb_id):07d}"
        return f"https://www.imdb.com/title/{clean_id}/"
    except (ValueError, TypeError):
        return None

def get_trailer_url(title):
    """Returns YouTube official trailer search link."""
    query = urllib.parse.quote(f"{title} official trailer")
    return f"https://www.youtube.com/results?search_query={query}"

def get_movie_poster(tmdb_id, title=None, api_key=None):
    """
    Fetches real movie poster URL from The Movie Database (TMDB) API if key is provided.
    Falls back to a curated, high-aesthetic cinema placeholder.
    """
    key = api_key or TMDB_API_KEY
    cache_key = f"{tmdb_id}_{key}"

    if cache_key in POSTER_CACHE:
        return POSTER_CACHE[cache_key]

    if key and tmdb_id and str(tmdb_id).lower() != 'nan':
        try:
            url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={key}&language=en-US"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    POSTER_CACHE[cache_key] = {
                        'url': poster_url,
                        'overview': data.get('overview', ''),
                        'backdrop': f"https://image.tmdb.org/t/p/w1280{data.get('backdrop_path')}" if data.get('backdrop_path') else None
                    }
                    return POSTER_CACHE[cache_key]
        except Exception:
            pass

    # High-quality fallback cinema card representation
    # Using clean cinema aesthetic poster placeholders
    safe_title = urllib.parse.quote(title or "Movie")
    fallback_url = f"https://placehold.co/400x600/181824/00DF81?text={safe_title}&font=montserrat"
    
    result = {
        'url': fallback_url,
        'overview': "Detailed plot overview available on IMDb & TMDB.",
        'backdrop': None
    }
    POSTER_CACHE[cache_key] = result
    return result
