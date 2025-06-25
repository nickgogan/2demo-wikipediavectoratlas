import os
import argparse
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loaded environment variables from {dotenv_path}")
    logger.info(f"Using embedding model: {os.getenv('EMBEDDING_MODEL')}")
else:
    logger.warning(f"No .env file found at {dotenv_path}, using system environment variables")

class EmbeddingGenerator:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding generator with a specified model.
        
        Args:
            model_name: Name of the model to use for generating embeddings.
                       If None, uses EMBEDDING_MODEL from environment variables.
        """
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model: {self.model_name} on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
            return embedding.cpu().numpy().tolist()
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
            return embeddings.cpu().numpy().tolist()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate embeddings for text using the specified model')
    parser.add_argument('text', nargs='?', help='Text to generate embedding for')
    parser.add_argument('--model', type=str, default=None, 
                        help='Model to use for generating embeddings (default: EMBEDDING_MODEL from .env)')
    parser.add_argument('--batch', action='store_true', 
                        help='Process multiple lines from stdin as a batch')
    parser.add_argument('--debug', action='store_true', 
                        help='Show debug logging and additional information')
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    
    # Initialize the embedding generator
    try:
        generator = EmbeddingGenerator(model_name=args.model)
    except Exception as e:
        logger.error(f"Failed to initialize the model: {str(e)}")
        return 1
    
    # Process input
    if args.batch:
        # Batch mode: read multiple lines from stdin
        import sys
        texts = [line.strip() for line in sys.stdin if line.strip()]
        if not texts:
            print("No input text provided", file=sys.stderr)
            return 1
        
        try:
            embeddings = generator.get_embeddings(texts)
            if args.debug:
                for text, embedding in zip(texts, embeddings):
                    print(f"Text: {text}")
                    print(f"Full embedding vector: {embedding}")
                    print(f"Embedding length: {len(embedding)}")
                    print("-" * 50)
            else:
                for embedding in embeddings:
                    print(embedding)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return 1
            
    elif args.text:
        # Single text mode
        try:
            embedding = generator.get_embedding(args.text)
            if args.debug:
                print(f"Text: {args.text}")
                print(f"Full embedding vector: {embedding}")
                print(f"Embedding length: {len(embedding)}")
            else:
                print(embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return 1
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
