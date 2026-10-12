# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-13
### Added
- Comprehensive automated PyTest test suite (`tests/test_recommender.py`) covering feature extraction, similarity monotonicity, and boundary criteria.
- Continuous Integration pipeline via GitHub Actions (`.github/workflows/ci.yml`).
- Containerization setup with production `Dockerfile` and `docker-compose.yml`.
- Standard MIT License and contribution documentation.

## [1.0.0] - 2026-10-12
### Added
- Initial release of Movie Recommendation System using MovieLens 100K dataset.
- TF-IDF feature extraction on genre multi-tokens and community tags.
- Pairwise Cosine Similarity calculation with sub-10ms top-N ranking.
- Modern cinematic Streamlit user interface with explainability engine.
- Exploratory data analysis (EDA) dashboard and theoretical mathematical formulation.
