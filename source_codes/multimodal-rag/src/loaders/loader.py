from llama_index.core import SimpleDirectoryReader
import chromadb
import json, os
from pathlib import Path
from datetime import datetime
import shutil
from dotenv import load_dotenv
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding 
from llama_index.core.schema import ImageNode
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from video_loader import process_video

VIDEO_FOLDER = 'data/video'
OUTPUT_FOLDER = "data"
OUTPUT_AUDIO_DIR = "data/audio" 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    print("\n1. Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path="chromadb-lli")

    # Create separate collections for text and images
    text_collection = client.get_or_create_collection("text_collection")
    image_collection = client.get_or_create_collection("image_collection")
    print("Collection initialized successfully")

    print("\n2. Setting up embedding function and storage...")
    text_embedding_function = OpenAIEmbedding(model="text-embedding-3-small")
    image_embedding_function = ClipEmbedding()

    # Create vector stores
    text_store = ChromaVectorStore(chroma_collection=text_collection, embedding=text_embedding_function)
    image_vstore = ChromaVectorStore(chroma_collection=image_collection, embedding=image_embedding_function)

    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_vstore)

    def load_data(folder_path):

        # loading video data by converting video to audio and transcribing
        try:
            process_video(
                VIDEO_FOLDER, 
                OUTPUT_FOLDER, 
                OUTPUT_AUDIO_DIR # Base path for audio files
            )
        except Exception as e:
            print(f"Error processing videos: {str(e)}")
            raise e
        
        print(f"\n3. Loading data from {folder_path}...")
        try:
            data = SimpleDirectoryReader(folder_path, 
                                        required_exts=['.txt', '.png', '.jpg', '.jpeg', '.mp4', '.mp3']).load_data()
            print(f"Loaded {len(data)} documents")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise e

        return data

    print("\n4. Creating vector store index...")
    index = MultiModalVectorStoreIndex.from_documents(
        documents=load_data(OUTPUT_FOLDER),
        storage_context=storage_context,
        text_embedding_function=text_embedding_function,
        image_embedding_function=image_embedding_function
    )
    print("   Index created successfully")

    print("\n5. Setting up query engine...")

    # # Initialize the multimodal LLM
    # openai_mm_llm = OpenAIMultiModal(
    #     model="gpt-4o",
    #     max_new_tokens=300
    # )
    # query_engine = index.as_query_engine(
    #     llm=openai_mm_llm,
    #     similarity_top_k=3
    # )
    # print("   Query engine ready")

    # print("\n6. Running example query...")
    # response = query_engine.query("Describe the images in the dataset")
    # print("   Query completed")

    print("\n7. Running image retrieval...")
    retriever = index.as_retriever(image_similarity_top_k=3)
    image_results = retriever.text_to_image_retrieve("What are the best technique of milk steaming, \
                                                     also find related images")
    print(f"   Retrieved {len(image_results)} images")

    retrieved_image = []
    retrieved_text = []
    for res_node in image_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            #display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)
    
    print(f"Retrieved {len(retrieved_image)} images and {retrieved_image}")
    print(f"Retrieved {len(retrieved_text)} texts and {retrieved_text}")

    #return retrieved_image, retrieved_text


if __name__ == "__main__":
    main()
