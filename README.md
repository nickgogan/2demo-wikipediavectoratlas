# Wikipedia Vector Search

This project processes Wikipedia data, generates vector embeddings using sentence transformers, and stores them in MongoDB for efficient search and retrieval.

## Features

- Download and process Wikipedia datasets
- Generate vector embeddings using state-of-the-art models
- Store and index data in MongoDB with vector search capabilities
- Configurable processing options (by records, size, or entire dataset)
- Modern configuration management using environment variables or `pyproject.toml`
- Support for both raw and pre-embedded document storage
- Toggle for embedding generation to optimize processing

## Prerequisites

- Python 3.8+
- MongoDB Atlas or local MongoDB instance (version 6.0+ for vector search)
- `uv` package manager (recommended) or `pip`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nickgogan/2demo-wikipediavectoratlas
   cd 2demo-wikipediavectoratlas
   ```

2. Set up a virtual environment (choose one method):

   **Using Python's built-in venv:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it (Linux/macOS)
   source venv/bin/activate
   
   # Or on Windows:
   # .\venv\Scripts\activate
   ```

   **Or using `uv` (faster alternative):**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   
   With `uv` (recommended):
   ```bash
   uv pip install -e ".[dev]"
   ```
   
   Or with `pip`:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

5. Copy the example environment file and update it with your configuration:
   ```bash
   cp .env.example .env
   # Edit .env with your values
   ```

## Configuration

### Environment Variables

Copy the example environment file and update the values:

```bash
cp .env.example .env
# Edit .env with your values
```

#### MongoDB Configuration
- `MONGO_URI`: MongoDB connection string (default: `mongodb://localhost:27017/`)
- `MONGO_DBNAME`: Database name (default: `DeveloperDay`)
- `MONGO_COLLNAME_RAW`: Collection for raw documents (default: `wikipedia`)
- `MONGO_COLLNAME_PREEMBEDDED`: Collection for documents with embeddings (default: `wikipedia_preembedded`)
- `RESET_COLLECTION`: Whether to drop existing collections on startup (default: `true`)
- `BATCH_SIZE`: Number of documents to process in each batch (default: `1000`)
- `VECTOR_FIELD_NAME`: Field name for storing embeddings (default: `embedding`)
- `GENERATE_EMBEDDINGS`: Whether to generate and store embeddings (default: `true`)

#### Dataset Configuration
- `DATASET_PATH`: Dataset path (default: `wikipedia`)
- `DATASET_NAME`: Dataset name/version (default: `20220301.en`)
- `DATASET_LANGUAGE`: Dataset language (default: `english`)
- `DATASET_DATE`: Dataset date (default: `20220301`)
- `DATA_MAX_BYTES`: Maximum data size to process in bytes (default: `11000000` ~11MB)
- `DATA_MAX_RECORDS`: Maximum number of records to process (default: `200`)
- `INDEX_BY`: Processing strategy: `ALL`, `BYTES`, or `RECORDS` (default: `ALL`)

#### Model Configuration
- `EMBEDDING_MODEL`: Sentence transformer model name (default: `sentence-transformers/all-mpnet-base-v2`)

## Usage

1. Configure your environment variables in `.env` or set them in your environment
2. Run the main script:
   ```bash
   python -m wikipedia_vector.main
   ```

The script will:
1. Connect to your MongoDB instance
2. Download and process the Wikipedia dataset
3. Store raw documents in the raw collection
4. If `GENERATE_EMBEDDINGS` is `true`:
   - Load the embedding model
   - Generate embeddings for each document
   - Store documents with embeddings in the pre-embedded collection

## Data Flow

1. **Data Loading**: The dataset is loaded in streaming mode to minimize memory usage
2. **Batch Processing**: Documents are processed in configurable batch sizes
3. **Dual Storage**:
   - Raw documents are stored in the raw collection
   - Documents with embeddings are stored in the pre-embedded collection (if enabled)
4. **Progress Tracking**: Progress is logged with document counts and data volumes

## Development

### Activating the Virtual Environment

When you return to the project, activate the virtual environment:

```bash
# If using Python's venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# If using uv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting
- **mypy** - Static type checking
- **pre-commit** - Git hooks for code quality

Run all checks:
```bash
pre-commit run --all-files
```

### Adding New Features

1. Create a new branch for your feature
2. Make your changes
3. Add tests if applicable
4. Update documentation
5. Submit a pull request

## License

MIT