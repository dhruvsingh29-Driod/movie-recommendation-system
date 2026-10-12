# Contributing to Movie Recommendation System

Thank you for your interest in contributing! We welcome contributions to improve the recommendation algorithms, user interface, and dataset feature processing.

## Development Workflow

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/<your-username>/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install pytest
   ```

3. **Run Existing Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Create a Feature Branch**
   ```bash
   git checkout -b feature/my-enhancement
   ```

5. **Commit with Conventional Commits**
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation updates
   - `test:` for test enhancements

6. **Submit a Pull Request**
   Ensure all tests pass before opening your PR.
