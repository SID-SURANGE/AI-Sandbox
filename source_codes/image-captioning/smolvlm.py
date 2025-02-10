"""
Image Captioning using SmolVLM Model

This module implements image captioning functionality using the SmolVLM-500M-Instruct model,
a lightweight vision-language model developed by HuggingFace. SmolVLM is a smaller, more efficient
alternative to larger vision-language models, making it suitable for resource-constrained environments.

Model Details:
- Name: SmolVLM-500M-Instruct
- Source: HuggingFace (HuggingFaceTB)
- Size: 500M parameters

API Key Requirements:
- HuggingFace API key (set in environment variables)
"""

# imports
import os
import argparse
from dotenv import load_dotenv
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load environment variables from .env file
load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = r".\data\sample.jpg"  # Path to the input image
CUSTOM_CACHE_DIR = "G:\LLM Models Download\hugging-face"
MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
MAX_NEW_TOKENS = 500

def generate_caption(image_path: str) -> str:
    """
    Generate a caption for the given image using the SmolVLM model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Generated caption for the image.
    """
    # Load and process image
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_CACHE_DIR)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
        cache_dir=CUSTOM_CACHE_DIR
    ).to(DEVICE)

    # Prepare input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"}
            ]
        },
    ]

    # Process inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

    # Generate caption
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate caption for an image.")
    parser.add_argument("--image_path", type=str, default=IMAGE_PATH,
                        help="Path to the image file. If not provided, a default image will be used.")
    args = parser.parse_args()

    caption = generate_caption(args.image_path)
    print(f"Generated caption: {caption}")
