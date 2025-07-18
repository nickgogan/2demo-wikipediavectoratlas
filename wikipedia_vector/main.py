"""Wikipedia Vector Embedding Processor

This module processes Wikipedia dataset, generates vector embeddings using sentence transformers,
and stores the results in MongoDB for efficient similarity search.
"""
import logging
import os
import signal
import sys
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
from tqdm import tqdm

from .config import AppConfig, IndexBy

# Configure logging to be tqdm-friendly
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())

# Global variable to signal exit
exit_requested = False

def handle_signal(signum, frame):
    """Signal handler to request a graceful exit."""
    global exit_requested
    if not exit_requested:
        logger.info("\nShutdown signal received. Finishing current batch before exiting...")
        exit_requested = True

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
    texts_to_embed = []
    docs_for_embedding = []
    
    try:
        # First, prepare all documents for processing
        for doc in batch:
            try:
                # Create document with only specified fields
                processed = {k: v for k, v in doc.items() if k in fields}
                raw_docs.append(processed)
                
                # Prepare for embedding if needed
                if embedded_collection is not None and encoder is not None and 'text' in processed:
                    texts_to_embed.append(processed['text'])
                    docs_for_embedding.append(processed)

            except Exception as e:
                logger.error(f"Error processing document: {e}")
        
        # Generate embeddings for the entire batch at once
        if texts_to_embed and encoder:
            try:
                embeddings = encoder.encode(texts_to_embed, show_progress_bar=False).tolist()
                
                for i, doc in enumerate(docs_for_embedding):
                    embedded_doc = doc.copy()
                    embedded_doc[vector_field] = embeddings[i]
                    embedded_docs.append(embedded_doc)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")

        # Insert raw documents
        raw_count = 0
        if raw_docs:
            try:
                result = raw_collection.insert_many(raw_docs, ordered=False)
                raw_count = len(result.inserted_ids)
                del result  # Clear the result to free memory
            except Exception as e:
                logger.error(f"Error inserting raw documents: {e}")
                raise
        
        # Insert embedded documents
        embedded_count = 0
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
        texts_to_embed.clear()
        docs_for_embedding.clear()

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
    
    # Determine the total number of documents for the progress bar
    total_docs = None
    if config.index_by == IndexBy.RECORDS and config.dataset.max_records:
        total_docs = config.dataset.max_records

    # Wrap the dataset with tqdm for a progress bar
    progress_bar = tqdm(dataset, desc="Processing dataset", unit="docs", total=total_docs)
    
    for doc in progress_bar:
        if exit_requested:
            logger.info("Exiting processing loop due to shutdown signal.")
            break

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
            
            # Update the progress bar with the latest stats
            progress_bar.set_postfix(
                batches=stats.batches, 
                docs=f"{stats.docs}",
                embedded=f"{stats.embedded}",
                size=size(stats.bytes_processed)
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
    
    # Process any remaining documents in the last batch if we didn't exit early
    if batch and not exit_requested:
        logger.info(f"Processing final batch of {len(batch)} documents...")
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
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

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
        dataset_str = config.dataset.path
        if config.dataset.name:
            dataset_str += f"/{config.dataset.name}"
        logger.info(f"Loading dataset: {dataset_str}")
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