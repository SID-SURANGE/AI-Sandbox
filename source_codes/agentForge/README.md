
# AgentForge: AI-Powered Assistant 

AgentForge is an intelligent assistant powered by AI that combines powerful web search capabilities with creative image generation. This application is built using the smolagents framework and Gradio UI.

## Features

- Model Selection: Choose between OpenAI GPT-4 and HuggingFace Llama models
- Web Search: Search for events, conferences, and gatherings worldwide
- Image Generation: Generate custom images from text descriptions
- File Upload: Support for PDF, DOCX, and TXT files

## Setup Instructions

1. Clone this repository:
```bash
git clone <your-repo-url>
cd AI-Sandbox/source_codes/agentForge

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     - OPENAI_API_KEY from [OpenAI Platform](https://platform.openai.com/api-keys)
     - HUGGING_FACE_HUB_TOKEN from [HuggingFace Settings](https://huggingface.co/settings/tokens)

4. Run the application:
```bash
python app.py
```

## Deploying to HuggingFace Spaces

1. Create a new Space on HuggingFace:
   - Go to [HuggingFace Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Set up the Space settings using the configuration in this README

2. Add your API keys as secrets in the Space:
   - Go to your Space's Settings > Variables and Secrets
   - Add OPENAI_API_KEY and HUGGING_FACE_HUB_TOKEN as secrets

3. Upload your code:
```bash
git add .
git commit -m "Initial commit"
git push
```

## Files Structure

- `app.py`: Main application file
- `Gradio_UI.py`: Gradio interface implementation
- `prompts.yaml`: Agent prompt templates
- `requirements.txt`: Python dependencies
- `.env.example`: Template for environment variables
- `tools/`: Directory containing agent tools
  - `final_answer.py`: Tool for generating final responses
  - `visit_webpage.py`: Tool for web page interactions
  - `web_search.py`: Tool for web searching

