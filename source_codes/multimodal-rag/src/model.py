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
CAPTION_MODEL_ID = "llava-v1.5-7b"
BASE_URL = "http://localhost:1234/v1"
API_KEY = "lm-studio"
TEMPERATURE = 0.2

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

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


def run_caption_model(image: Image.Image, file_path: str) -> str:
    """
    Generate a descriptive caption for the given PIL Image using LLM.
    Args:
        image: PIL Image object
    Returns:
        str: Generated caption describing the image
    """
    try:
        # Convert PIL Image to base64 if required by the model
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Truncate the input context if it exceeds the model's maximum context length
        max_context_length = 4000
        user_content = f"Describe the image focusing on the objects and their interactions if any. Image data: data:image/png;base64,{img_str}"
        if len(user_content) > max_context_length:
            user_content = user_content[:max_context_length]

        # Use the image directly if the model supports it
        completion = client.chat.completions.create(
            model=CAPTION_MODEL_ID,
            messages=[
                {"role": "system", "content": """You are an expert caption generator focused on providing descriptive captions that emphasize both the objects identified in the image and any interactions between them. When analyzing an image:
                1. Identify the main objects present.
                2. Describe the interactions, actions, or relationships between these objects if any.
                3. Provide a concise and natural description without technical jargon.
                4. Ensure the caption directly reflects what is visible in the image."""},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
        )

        caption = completion.choices[0].message.content
        print(f"\nGenerated caption for file {file_path}: {caption}")
        logger.info(f"\nGenerated caption {file_path}: {caption}")

        return caption

    except Exception as e:
        logger.error(f"\nError generating image caption: {str(e)}")
        return "Error generating image description"
