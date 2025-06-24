# Wikipedia Vector Search

This project processes Wikipedia data, generates vector embeddings using sentence transformers, and stores them in MongoDB for efficient search and retrieval.

## Features

- Download and process Wikipedia datasets
- Generate vector embeddings using state-of-the-art models
- Store and index data in MongoDB with vector search capabilities
- Configurable processing options (by records, size, or entire dataset)
- Modern configuration management using environment variables or `pyproject.toml`

## Prerequisites

- Python 3.8+
- MongoDB Atlas or local MongoDB instance
- `uv` package manager (recommended) or `pip`

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd wikipedia-vector
   ```

2. Install dependencies using `uv` (recommended):
   ```bash
   uv pip install -e ".[dev]"
   ```

   Or using `pip`:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Setup

This project uses several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Static type checking
- **pre-commit** - Git hooks for code quality

### Pre-commit Hooks

Pre-commit hooks are automatically installed when you run `pre-commit install` after installing the development dependencies. These hooks will run automatically before each commit to ensure code quality.

To run all checks manually:
```bash
pre-commit run --all-files
```

## Configuration

### Option 1: Using `pyproject.toml` (Recommended)

Edit the `[tool.uv.env]` section in `pyproject.toml` with your configuration:

```toml
[tool.uv.env]
# MongoDB Configuration
MONGO_URI = "mongodb+srv://<username>:<password>@<clustername>.<projecthash>.mongodb.net/?retryWrites=true"
MONGO_DBNAME = "DevDay"
MONGO_COLLNAME = "wikipedia"
RESET_COLLECTION = "true"
BATCH_SIZE = "1000"

# Dataset Configuration
DATASET_PATH = "wikipedia"
DATASET_NAME = "20220301.en"
DATASET_LANGUAGE = "english"
DATASET_DATE = "20220301"

# Processing Configuration
INDEX_BY = "ALL"  # Options: ALL, BYTES, RECORDS
DATA_MAX_BYTES = "11000000"  # 11MB (used when INDEX_BY=BYTES)
DATA_MAX_RECORDS = "200"     # Used when INDEX_BY=RECORDS

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Option 2: Using Environment Variables

Copy the example environment file and update the values:

```bash
cp .env.example .env
# Edit .env with your values
```

### Option 3: System Environment Variables

Set the variables directly in your shell or deployment environment.

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. The script will:
   - Connect to your MongoDB instance
   - Download and process the Wikipedia dataset
   - Generate vector embeddings for each document
   - Store the documents in MongoDB

## Indexing Strategies

- `ALL`: Process the entire dataset
- `BYTES`: Process until reaching the specified data volume
- `RECORDS`: Process a specific number of records

## Project Structure

- `main.py`: Main application entry point
- `config.py`: Configuration management using Python dataclasses
- `.env.example`: Example environment variables
- `pyproject.toml`: Project metadata and configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration\

## License

MIT
