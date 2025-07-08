import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal
from dotenv import load_dotenv
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f"No .env file found at {dotenv_path}, using system environment variables")

class IndexBy(str, Enum):
    ALL = 'ALL'
    BYTES = 'BYTES'
    RECORDS = 'RECORDS'

@dataclass
class MongoConfig:
    uri: str
    db_name: str
    collection_raw: str
    collection_preembedded: str
    reset_collection: bool
    batch_size: int
    vector_field: str
    
    @classmethod
    def from_env(cls) -> 'MongoConfig':
        return cls(
            uri=os.getenv('MONGO_URI', 'mongodb://localhost:27017/'),
            db_name=os.getenv('MONGO_DBNAME', 'DeveloperDay'),
            collection_raw=os.getenv('MONGO_COLLNAME_RAW', 'wikipedia'),
            collection_preembedded=os.getenv('MONGO_COLLNAME_PREEMBEDDED', 'wikipedia_preembedded'),
            reset_collection=os.getenv('RESET_COLLECTION', 'true').lower() == 'true',
            batch_size=int(os.getenv('BATCH_SIZE', '1000')),
            vector_field=os.getenv('VECTOR_FIELD_NAME', 'embedding')
        )

@dataclass
class DatasetConfig:
    path: str
    name: str
    max_bytes: Optional[float]
    max_records: Optional[int]

    @classmethod
    def from_env(cls) -> 'DatasetConfig':
        def clean_env_value(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            # Remove any trailing comments and whitespace
            return value.split('#')[0].strip() if value else None
            
        max_bytes = clean_env_value(os.getenv('DATA_MAX_BYTES'))
        max_records = clean_env_value(os.getenv('DATA_MAX_RECORDS'))
        
        index_by = IndexBy(os.getenv('INDEX_BY', 'ALL').upper())
        
        if index_by == IndexBy.BYTES:
            logger.info(f"Loading config - BYTES mode with max_bytes: {max_bytes}")
        elif index_by == IndexBy.RECORDS:
            logger.info(f"Loading config - RECORDS mode with max_records: {max_records}")
        else:
            logger.info("Loading config - ALL mode (no limits)")
            
        return cls(
            path=os.getenv('DATASET_PATH', 'wikipedia'),
            name=os.getenv('DATASET_NAME', '20220301.en'),
            max_bytes=float(max_bytes) if max_bytes else None,
            max_records=int(max_records) if max_records else None
        )

@dataclass
class ModelConfig:
    name: str
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        return cls(
            name=os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        )

@dataclass
class AppConfig:
    mongo: MongoConfig
    dataset: DatasetConfig
    model: ModelConfig
    index_by: IndexBy
    generate_embeddings: bool = True  # Default to True for backward compatibility
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        index_by_str = os.getenv('INDEX_BY', 'ALL').upper()
        try:
            index_by = IndexBy(index_by_str)
            logger.info(f"Using INDEX_BY: {index_by}")
        except ValueError as e:
            valid_values = [e.value for e in IndexBy]
            raise ValueError(
                f"Invalid INDEX_BY value: {index_by_str}. "
                f"Must be one of {valid_values}"
            ) from e

        return cls(
            mongo=MongoConfig.from_env(),
            dataset=DatasetConfig.from_env(),
            model=ModelConfig.from_env(),
            index_by=index_by,
            generate_embeddings=os.getenv('GENERATE_EMBEDDINGS', 'true').lower() == 'true'
        )
