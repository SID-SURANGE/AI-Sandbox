"""
Image Captioning using Llama 3.2 Vision Model

This module implements image captioning functionality using Meta's Llama 3.2 Vision model,
accessed through the Together AI platform. The model combines advanced vision understanding
with natural language generation capabilities to provide detailed image descriptions.

Model Details:
- Name: Llama-3.2-11B-Vision-Instruct-Turbo
- Source: Meta AI (accessed via Together AI platform)
- Size: 11B parameters

API Key Requirements:
- Together AI API key (set in environment variables)
"""

# Import required libraries for image processing, API interaction, and environment variables
import os, base64
import argparse
from together import Together
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants and Configuration
IMAGE_PATH = r".\data\sample.jpg"  # Path to the input image
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"  # Model identifier
MAX_TOKENS = 700  # Maximum number of tokens in the response
TEMPERATURE = 0.18  # Controls randomness in generation (lower = more deterministic)
TOP_P = 0.5  # Nucleus sampling parameter
TOP_K = 10  # Top-k sampling parameter
REP_PENALTY = 1  # Repetition penalty
STOP_SEQUENCES = ["<|eot_id|>", "<|eom_id|>"]  # Sequences to stop generation

# Initialize the Together API client
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def image_caption(image_path: str):
    """
    Generate a caption for the provided image using the Llama Vision model.
    
    Args:
        image_path (str): Path to the image file to be captioned
    
    Returns:
        None: Prints the generated caption and keywords
    """
    # Read and encode the image to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Define the prompt template for image captioning
    prompt = """Describe the image in a short, concise and descriptive manner.
                Additionally, list 5 major keywords that would aptly describe the image details.
                Response format should strictly be a JSON object in a single paragraph without newlines,
                for example: 
                    {"response":"<image description>",
                    "keywords":["keyword1","keyword2","keyword3","keyword4","keyword5"]}
             """
        
    # Create the image URL with base64 encoding
    image_url = f"data:image/png;base64,{encoded_image}"

    # Prepare the API request payload
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    # Generate the caption using the model
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REP_PENALTY,
        stop=STOP_SEQUENCES,
        stream=True
    )

    # Process the streaming response
    output = ""
    for token in response:
        if hasattr(token, 'choices') and token.choices:
            try:
                output += token.choices[0].delta.content
            except (IndexError, AttributeError) as error:
                print("Error extracting token content:", error)
    print("Final output:", output)

# Script entry point
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate caption for an image.")
    parser.add_argument("--image_path", type=str, default=IMAGE_PATH,
                        help="Path to the image file. If not provided, a default image will be used.")
    args = parser.parse_args()

    image_caption(args.image_path)