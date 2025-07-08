"""Wikipedia Vector Embedding Processor

This module processes Wikipedia dataset, generates vector embeddings using sentence transformers,
and stores the results in MongoDB for efficient similarity search.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import PyMongoError
from datasets import load_dataset, IterableDataset
from sentence_transformers import SentenceTransformer
from hurry.filesize import size
from pympler import asizeof
from dataclasses import dataclass, field
import gc

from .config import AppConfig, IndexBy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TODOs
# 1. Address potential memory leak issues

@dataclass
class ProcessingStats:
    """Tracks processing statistics and metrics."""
    batches: int = 0
    docs: int = 0
    embedded: int = 0
    bytes_processed: int = 0
    
    def add_batch(self, docs_processed: int, docs_embedded: int, bytes_processed: int) -> None:
        """Update statistics for a processed batch."""
        self.batches += 1
        self.docs += docs_processed
        self.embedded += docs_embedded
        self.bytes_processed += bytes_processed

def create_mongo_client(uri: str) -> MongoClient:
    """Create and return a MongoDB client with error handling."""
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, server_api=ServerApi('1'))
        # Test the connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except PyMongoError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def process_batch(
    batch: List[Dict[str, Any]],
    raw_collection: Collection,
    embedded_collection: Optional[Collection],
    encoder: Optional[SentenceTransformer],
    vector_field: str,
    fields: List[str]
) -> tuple[int, int]:
    """Process a batch of documents and insert them into MongoDB collections."""
    if not batch:
        return 0, 0
    
    raw_docs = []
    embedded_docs = []
    
    try:
        # Process documents with minimal memory retention
        for doc in batch:
            try:
                # Create document with only specified fields
                processed = {k: v for k, v in doc.items() if k in fields}
                raw_docs.append(processed)
                
                # Generate embedding if needed
                if embedded_collection is not None and encoder is not None and 'text' in processed:
                    try:
                        # Create embedding without keeping extra references
                        embedding = encoder.encode(processed['text']).tolist()
                        embedded_doc = {k: v for k, v in processed.items()}
                        embedded_doc[vector_field] = embedding
                        embedded_docs.append(embedded_doc)
                        # Clear references
                        del embedded_doc, embedding
                    except Exception as e:
                        logger.error(f"Error generating embedding: {e}")
                
                # Clear references
                del processed
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        # Insert raw documents
        if raw_docs:
            try:
                result = raw_collection.insert_many(raw_docs, ordered=False)
                raw_count = len(result.inserted_ids)
                del result  # Clear the result to free memory
            except Exception as e:
                logger.error(f"Error inserting raw documents: {e}")
                raise
        
        # Insert embedded documents
        if embedded_docs and embedded_collection is not None:
            try:
                result = embedded_collection.insert_many(embedded_docs, ordered=False)
                embedded_count = len(result.inserted_ids)
                del result  # Clear the result to free memory
            except Exception as e:
                logger.error(f"Error inserting embedded documents: {e}")
        
        return raw_count, embedded_count
        
    finally:
        # Explicitly clear all references
        raw_docs.clear()
        embedded_docs.clear()

def process_dataset(
    dataset: IterableDataset,
    raw_collection: Collection,
    embedded_collection: Optional[Collection],
    config: AppConfig
) -> None:
    """Process the dataset and store in MongoDB."""
    encoder = None
    if embedded_collection is not None:
        logger.info(f"Loading model: {config.model.name}")
        encoder = SentenceTransformer(
            config.model.name,
        )
    
    batch = []
    batch_size = 0
    stats = ProcessingStats()
    
    for doc in dataset:
        doc_size = asizeof.asizeof(doc)
        
        # Add to batch
        batch.append(doc)
        batch_size += doc_size
        
        # Check if we've reached the batch size
        if (len(batch) >= config.mongo.batch_size or 
            (config.index_by == IndexBy.BYTES and 
             batch_size >= (config.dataset.max_bytes or float('inf')))):
            
            # Process batch
            raw_count, embedded_count = process_batch(
                batch,
                raw_collection,
                embedded_collection,
                encoder,
                config.mongo.vector_field,
                ['chunk_id', 'text']
            )
            
            # Update statistics
            stats.add_batch(raw_count, embedded_count, batch_size)
            
            # Log progress
            logger.info(
                f"Processed batch {stats.batches}: "
                f"{raw_count} docs (total: {stats.docs} docs, "
                f"{stats.embedded} embedded, {size(stats.bytes_processed)} processed)"
            )
            
            # Clear batch and trigger garbage collection to free memory
            batch.clear()
            batch_size = 0
            gc.collect()
            
            # Check if we've reached max records
            if (config.index_by == IndexBy.RECORDS and 
                config.dataset.max_records and 
                stats.docs >= config.dataset.max_records):
                logger.info(f"Reached max records: {stats.docs}")
                break
            
            # Check if we've reached max bytes
            if (config.index_by == IndexBy.BYTES and 
                config.dataset.max_bytes and 
                stats.bytes_processed >= config.dataset.max_bytes):
                logger.info(f"Reached max bytes: {size(stats.bytes_processed)}")
                break
    
    # Process any remaining documents in the last batch
    if batch:
        raw_count, embedded_count = process_batch(
            batch,
            raw_collection,
            embedded_collection,
            encoder,
            config.mongo.vector_field,
            ['chunk_id', 'text']
        )
        stats.add_batch(raw_count, embedded_count, batch_size)
        batch.clear()
        gc.collect()
        
    logger.info(
        f"Processing complete. Processed {stats.batches} batches with "
        f"{stats.docs} total documents ({stats.embedded} with embeddings), "
        f"totaling {size(stats.bytes_processed)}"
    )

def main() -> None:
    """Main function to process the Wikipedia dataset."""
    try:
        # Load configuration
        config = AppConfig.from_env()
        logger.info(f"Starting processing with config: {config}")
        
        # Setup MongoDB
        mongo_client = create_mongo_client(config.mongo.uri)
        db = mongo_client.get_database(config.mongo.db_name)
        
        # Reset collections if needed
        if config.mongo.reset_collection:
            db[config.mongo.collection_raw].drop()
            logger.info(f"Dropped collection: {config.mongo.db_name}.{config.mongo.collection_raw}")
            
            if hasattr(config.mongo, 'collection_preembedded'):
                db[config.mongo.collection_preembedded].drop()
                logger.info(f"Dropped collection: {config.mongo.db_name}.{config.mongo.collection_preembedded}")
        
        # Get collections
        raw_collection = db[config.mongo.collection_raw]
        embedded_collection = None
        
        # Only create pre-embedded collection if we have an embedding model and flag is True
        if config.generate_embeddings and hasattr(config.mongo, 'collection_preembedded'):
            logger.info("Embedding generation is enabled")
            embedded_collection = db[config.mongo.collection_preembedded]
        else:
            logger.info("Embedding generation is disabled")
        
        # Load dataset
        logger.info(f"Loading dataset: {config.dataset.path}/{config.dataset.name}")
        dataset: IterableDataset = load_dataset(
            config.dataset.path, 
            config.dataset.name, 
            split='train', 
            streaming=True,
            trust_remote_code=True
        )
        
        # Only apply max_records limit if INDEX_BY is RECORDS
        if config.index_by == IndexBy.RECORDS and config.dataset.max_records is not None:
            logger.info(f"Limiting to {config.dataset.max_records} records")
            dataset = dataset.take(config.dataset.max_records)
        
        # Process the dataset
        process_dataset(dataset, raw_collection, embedded_collection, config)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'mongo_client' in locals():
            mongo_client.close()

if __name__ == "__main__":
    main()