# Standard library imports
import logging
from typing import Dict
from io import BytesIO
import base64

# Third-party imports
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
# OpenAI imports
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "hermes-3-llama-3.2-3b"
CAPTION_MODEL_ID = "hermes-3-llama-3.2-3b"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
TEMPERATURE = 0.2

def run_llm(user_query:str, context: str) -> JSONResponse:
    """
    Function to interact with the LLM (Language Model) and fetch a response.

    Args:
        query (str): The user's query to be processed by the LLM.

    Returns:
        JSONResponse: A JSON response containing the LLM's output.
    """
    try:
        # Initialize OpenAI client with local server configuration
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        
        # Define the chat completion request
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": """You are an AI assistant that helps analyze and synthesize information from multiple documents. 
                When provided with context from multiple documents and a user query:
                1. Carefully analyze all provided document snippets
                2. Identify relevant information from each document that relates to the user's query
                3. Synthesize a comprehensive response that combines insights from all provided documents
                4. Ensure the response directly addresses the user's question
                5. If there are any contradictions between documents, acknowledge them in your response
                6. If the user's query is not clear, ask for clarification
                7. Response strictly must not contains any reference to the documents by name or number like 'based on the provided documents' or 'from the documents' """},
                
                {"role": "user", "content": f"""User Query: {user_query}

                Retrieved Documents:
                {context}

                Please provide a comprehensive response based on the information from all these documents while directly addressing the user's query."""}
            ],
            temperature=TEMPERATURE,
        )
        
        # Extract the response from the LLM's output
        response_message = completion.choices[0].message.content
        
        # Log the response for debugging purposes
        logger.info(f"LLM Response: {response_message}")
        
        return response_message
    
    except Exception as e:
        # Log the error and raise an HTTPException for FastAPI to handle
        logger.error(f"Error in LLM completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error in LLM completion: {str(e)}")


def run_caption_model(image: Image.Image) -> str:
    """
    Generate a descriptive caption for the given PIL Image using LLM.
    Args:
        image: PIL Image object
    Returns:
        str: Generated caption describing the image
    """
    try: 
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        
        # Convert PIL Image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": """You are an expert at analyzing images and providing detailed descriptions. 
                When given an image:
                1. Describe the main subjects and objects clearly
                2. Note important visual details (colors, composition, setting)
                3. Keep descriptions concise yet informative (1-2 sentences)
                4. Focus on factual observations, avoid subjective interpretations
                5. Use natural, flowing language suitable for search and retrieval"""},
                
                {"role": "user", "content": [
                    {"type": "text", "text": "Please provide a clear and concise description of this image that captures its key visual elements."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]}
            ],
            temperature=0.7,
        )
        
        caption = completion.choices[0].message.content
        logger.info(f"Generated caption: {caption}")
        
        return caption
        
    except Exception as e:
        logger.error(f"Error generating image caption: {str(e)}")
        return "Error generating image description"
