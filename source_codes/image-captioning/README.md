# Image Captioning

This project implements image captioning functionality using two different vision-language models:
1. Meta's Llama 3.2 Vision (via Together AI)
2. HuggingFace's SmolVLM

## Models

### Llama 3.2 Vision
- **Size**: 11B parameters
- **Provider**: Meta AI (accessed through Together AI)
- **Features**: Multi-modal capabilities with structured JSON output
- **File**: `llama3.2-vision.py`

### SmolVLM
- **Size**: 500M parameters
- **Provider**: HuggingFace (HuggingFaceTB)
- **Features**: Lightweight model with Flash Attention 2.0 support
- **File**: `smolvlm.py`

## Setup

### Prerequisites
1. Python 3.x
2. CUDA-capable GPU (optional, for faster inference)
3. Required API keys:
   - Together AI API key
   - HuggingFace API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-captioning
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Setup
1. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API keys:
   ```
   TOGETHER_API_KEY=your_together_api_key
   HUGGING_FACE_HUB_TOKEN=your_huggingface_token
   ```

## Usage

To use the image captioning models, run the Python files directly:

1. For Llama 3.2 Vision:
   ```bash
   python llama3_2_vision.py data/your_image.jpg
   ```

2. For SmolVLM:
   ```bash
   python smolvlm.py data/your_image.jpg
   ```

Replace `your_image.jpg` with the actual filename of your image in the `data` directory.

## Input Data

Place your images in the `data` directory. Supported image formats:
- JPEG/JPG
- PNG
- Other common image formats supported by PIL


## Note
- Ensure you have sufficient disk space for model downloads
- GPU is recommended but not required
- <font color="red">The current implementation does not include comprehensive error handling. Users are encouraged to add their own error handling when cloning the repository or to ensure correct parameters are used when running the scripts.</font>
