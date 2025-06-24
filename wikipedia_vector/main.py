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

from .config import AppConfig, IndexBy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """Process a batch of documents and insert them into MongoDB collections.
    
    Args:
        batch: List of documents to process
        raw_collection: MongoDB collection for raw documents
        embedded_collection: MongoDB collection for documents with embeddings
        encoder: SentenceTransformer instance for generating embeddings
        vector_field: Field name to store the embedding vector
        fields: List of fields to keep in the documents
    
    Returns:
        Tuple[int, int]: Number of documents inserted into raw and embedded collections
    """
    if not batch:
        return 0, 0
    
    raw_docs = []
    embedded_docs = []
    
    try:
        for doc in batch:
            try:
                # Create document with only specified fields
                processed = {k: v for k, v in doc.items() if k in fields}
                
                # Always add to raw collection
                raw_docs.append(processed)
                
                # Add to embedded collection if encoder is available and document has text
                if embedded_collection is not None and encoder is not None and 'text' in processed:
                    try:
                        # Create a copy to avoid modifying the original
                        embedded_doc = processed.copy()
                        # Generate and add embedding
                        embedding = encoder.encode(embedded_doc['text']).tolist()
                        embedded_doc[vector_field] = embedding
                        embedded_docs.append(embedded_doc)
                        # Clear references
                        del embedded_doc
                    except Exception as e:
                        logger.error(f"Error generating embedding: {e}")
                
                # Clear the processed doc reference
                del processed
                
            except Exception as e:
                logger.error(f"Error processing document in batch: {e}")
        
        # Insert into collections
        raw_count = 0
        embedded_count = 0
        
        # Insert raw documents
        if raw_docs:
            try:
                result = raw_collection.insert_many(raw_docs, ordered=False)
                raw_count = len(result.inserted_ids)
            except Exception as e:
                logger.error(f"Error inserting raw documents: {e}")
                raise
        
        # Insert embedded documents if any
        if embedded_docs and embedded_collection is not None:
            try:
                result = embedded_collection.insert_many(embedded_docs, ordered=False)
                embedded_count = len(result.inserted_ids)
            except Exception as e:
                logger.error(f"Error inserting embedded documents: {e}")
                # Don't raise here to ensure we still return the raw count
        
        return raw_count, embedded_count
        
    finally:
        # Explicitly clear lists to free memory
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
            
            # Calculate actual size of this batch
            batch_bytes = sum(asizeof.asizeof(doc) for doc in batch)
            
            # Process batch
            raw_count, embedded_count = process_batch(
                batch,
                raw_collection,
                embedded_collection,
                encoder,
                config.mongo.vector_field,
                ['id', 'text', 'title', 'url']
            )
            
            # Update statistics with actual batch size
            stats.add_batch(raw_count, embedded_count, batch_bytes)
            
            # Log progress
            logger.info(
                f"Processed batch {stats.batches}: "
                f"{raw_count} docs (total: {stats.docs} docs, "
                f"{stats.embedded} embedded, {size(stats.bytes_processed)} processed)"
            )
            
            batch = []
            batch_size = 0
            
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
        batch_bytes = sum(asizeof.asizeof(doc) for doc in batch)
        raw_count, embedded_count = process_batch(
            batch,
            raw_collection,
            embedded_collection,
            encoder,
            config.mongo.vector_field,
            ['id', 'text', 'title', 'url']
        )
        stats.add_batch(raw_count, embedded_count, batch_bytes)
        
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