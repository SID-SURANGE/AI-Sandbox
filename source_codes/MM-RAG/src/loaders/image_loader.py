# third-party imports
from PIL import Image
import uuid
from pathlib import Path
import logging

# local imports
from utils.config import collection, image_model, FIXED_DIMENSION
from utils.helpers import adjust_embedding_dimension
from src.model import run_caption_model

logger = logging.getLogger(__name__)


def process_image(file_path):
    """
    Process an image file, generate embeddings, and store in ChromaDB.

    Args:
        file_path (str): Path to the image file.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing image file: {file_path}")
        
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Try opening and validating image
        try:
            image = Image.open(file_path)
            image.verify()  # Verify it's a valid image
            image = Image.open(file_path)  # Reopen after verify
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")

        # Generate image embedding
        try:
            embedding = image_model.encode(image)
            adjusted_embedding = adjust_embedding_dimension(embedding, FIXED_DIMENSION)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image embedding: {str(e)}")

        # Generate image description
        try:
            description = run_caption_model(image)
        except Exception as e:
            logger.warning(f"Failed to generate image caption: {str(e)}")
            description = ""

        # Add to ChromaDB
        try:
            collection.add(
                documents=[str(file_path)],
                embeddings=[adjusted_embedding],
                metadatas=[{
                    "type": "image",
                    "filename": Path(file_path).name,
                    "image_path": str(file_path),
                    "description": description,
                    "metadata": {
                        "size": image.size,
                        "format": image.format,
                        "mode": image.mode,
                    }
                }],
                ids=[f"img_{uuid.uuid4()}"]
            )
            print(f"Image file '{file_path}' processed and added to ChromaDB.")
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to add image to ChromaDB: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")
        print(f"Failed to process image {file_path}: {str(e)}")
        return False

    finally:
        # Clean up if needed
        if 'image' in locals():
            image.close()
