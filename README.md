# Wikipedia Vector Search

This project is a data processing pipeline that fetches data from a Hugging Face dataset, generates vector embeddings for the text content, and stores the results in a MongoDB Atlas collection. It is designed to be configurable and scalable for handling large datasets.

## Features

- **Data Ingestion**: Loads datasets directly from Hugging Face, supporting streaming for large datasets.
- **Vector Embeddings**: Uses `sentence-transformers` to generate high-quality vector embeddings for text data.
- **MongoDB Integration**: Stores both the raw text and the vector embeddings in MongoDB Atlas, ready for vector search.
- **Configurable**: All key parameters, such as database connections, dataset names, and model names, are managed through a `.env` file.
- **Memory Efficient**: Designed with memory management in mind, including batch processing and explicit garbage collection to handle large-scale data processing.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- A MongoDB Atlas account (a free cluster is sufficient).

### 1. Clone the Repository

```bash
git clone https://github.com/nickgogan/2demo-wikipediavectoratlas.git
cd 2demo-wikipediavectoratlas
```

### 2. Set Up a Virtual Environment

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The application uses a `.env` file to manage configuration settings. To get started, copy the example file:

```bash
cp .env.example .env
```

Next, open the `.env` file and update the `MONGO_URI` with your MongoDB Atlas connection string. You can get this from the Atlas UI.

```dotenv
# .env
MONGO_URI=mongodb+srv://<username>:<password>@<clustername>.<projecthash>.mongodb.net/?retryWrites=true
# ... other settings
```

## Usage

Once the setup and configuration are complete, you can run the data processing pipeline with the following command from the project's root directory:

```bash
python -m wikipedia_vector.main
```

The script will begin fetching the dataset, generating embeddings, and ingesting the data into your specified MongoDB collection.

## Configuration Reference

The behavior of the script can be customized through the following environment variables in the `.env` file:

| Variable | Description |
|---|---|
| `MONGO_URI` | **Required.** Your MongoDB Atlas connection string. |
| `MONGO_DBNAME` | The name of the database to use. Defaults to `DeveloperDay`. |
| `MONGO_COLLNAME_RAW` | The name of the collection for raw documents. Defaults to `wikipedia`. |
| `MONGO_COLLNAME_PREEMBEDDED` | The name of the collection for documents with embeddings. Defaults to `wikipedia_preembedded`. |
| `RESET_COLLECTION` | If `true`, drops the collections before starting. Defaults to `true`. |
| `BATCH_SIZE` | The number of documents to process in each batch. Defaults to `1000`. |
| `GENERATE_EMBEDDINGS` | If `true`, generates and stores embeddings. Defaults to `true`. |
| `VECTOR_FIELD_NAME` | The name of the field for storing embedding vectors. Defaults to `embedding`. |
| `DATASET_PATH` | The path to the dataset on Hugging Face. Defaults to `AIatMongoDB/cosmopedia-wikihow-chunked`. |
| `DATASET_NAME` | The specific configuration of the dataset (if any). Defaults to empty. |
| `INDEX_BY` | The strategy for limiting data processing (`ALL`, `BYTES`, or `RECORDS`). Defaults to `ALL`. |
| `DATA_MAX_BYTES` | The maximum number of bytes to process (used if `INDEX_BY=BYTES`). |
| `DATA_MAX_RECORDS` | The maximum number of records to process (used if `INDEX_BY=RECORDS`). |
| `EMBEDDING_MODEL` | The name of the `sentence-transformers` model to use. Defaults to `sentence-transformers/all-mpnet-base-v2`. |

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