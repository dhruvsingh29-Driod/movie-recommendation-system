# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-10-13
### Added
- Item-based Collaborative Filtering engine (`src/collaborative.py`) utilizing sparse user-item interaction matrix.
- Hybrid Recommendation engine blending TF-IDF content similarity with collaborative signals.
- Offline evaluation benchmark suite (`src/evaluate.py`) calculating Precision@K, Recall@K, MRR, and Catalog Coverage.
- Production REST API microservice (`api.py`) exposing `/api/v1/recommend`, `/api/v1/movies`, `/api/v1/evaluate`, and interactive documentation at `/docs`.
- Expanded test suite with 9 passing tests in `tests/test_collaborative.py`.

## [1.1.0] - 2026-10-13
### Added
- Comprehensive automated PyTest test suite (`tests/test_recommender.py`) covering feature extraction, similarity monotonicity, and boundary criteria.
- Containerization setup with production `Dockerfile` and `docker-compose.yml`.
- Standard MIT License and contribution documentation.

## [1.0.0] - 2026-10-12
### Added
- Initial release of Movie Recommendation System using MovieLens 100K dataset.
- TF-IDF feature extraction on genre multi-tokens and community tags.
- Pairwise Cosine Similarity calculation with sub-10ms top-N ranking.
- Modern cinematic Streamlit user interface with explainability engine.
- Exploratory data analysis (EDA) dashboard and theoretical mathematical formulation.
