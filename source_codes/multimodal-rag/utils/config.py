from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

# Initialize cross-encoder model for more accurate result ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Define collection schema
collection_schema = {
    "text": {
        "document": str,
        "embedding": list,
        "metadata": {
            "type": "text",
            "filename": str
        }
    },
    "image": {
        "image_path": str,
        "embedding": list,
        "metadata": {
            "type": "image",
            "filename": str,
            "description": str
        },
    },
    "video": {
        "video_path": str,
        "embedding": list,
        "metadata": {
            "type": "video",
            "filename": str,
            "duration": float,
            "frames": {
                "timestamp": float,
                "frame_embedding": list,
                "frame_description": str
            },
            "audio": {
                "transcript": str,
                "transcript_embedding": list,
                "segments": [{
                    "start": float,
                    "end": float,
                    "text": str
                }]
            }
        }
    },
    "audio": {
        "audio_path": str,
        "embedding": list,
        "metadata": {
            "type": "audio",
            "filename": str,
            "description": str
        }
    }
}

# Initialize ChromaDB Client
client = PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(name="multimodal_data_new")

# Text splitter configuration
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Model Configuration
text_model = SentenceTransformer("all-mpnet-base-v2")
image_model = SentenceTransformer("clip-ViT-L-14")

# Constants
FIXED_DIMENSION = 512
