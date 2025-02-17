> **Note:** The `dev` branch is the active branch for the latest updates. Please switch to that branch for the most recent code updates.

# AI-Sandbox
My personal AI sandbox for learning, experimenting, and prototyping different AI models and techniques.

## Project Summary

| Project | Description | Key Technologies |
|---------|------------|------------------|
| [Multimodal RAG](./source_codes/multimodal-rag/) | A barista chatbot with multimodal retrieval-augmented generation for handling menu items and coffee recipes | Streamlit, FastAPI, Llama, ChromaDB, CLIP |
| [Chat with GitHub Repo](./source_codes/chat-with-gitrepo/) | Interactive chat interface for exploring and querying GitHub repositories | Streamlit, ChromaDB, spaCy, LlamaIndex |
| [AgentForge](./source_codes/agentforge/) | AI assistant combining web search and image generation capabilities | Gradio, smolagents, GPT-4, Llama |
| [Image Captioning](./source_codes/image-captioning/) | AI-powered image captioning using Llama Vision model | Together AI, Llama-3.2-Vision, Pydantic |

## Projects

### 1. [Multimodal RAG](./source_codes/multimodal-rag/)
A barista chatbot built with multimodal retrieval-augmented generation capabilities.

**Features:**
- Answers questions about menu items and coffee recipes
- Handles multiple content types (text, images, video)
- Admin mode for data management
- User-friendly Streamlit interface

**Tech Stack:**
- Frontend: Streamlit
- Backend: FastAPI
- LLM: Hermes-Llama-3.2-3b-instruct
- Vector Database: ChromaDB
- Image/Text Embeddings: CLIP, SentenceTransformers

### 2. [Chat with GitHub Repo](./source_codes/chat-with-gitrepo/)
A Streamlit application that enables conversations with GitHub repositories.

**Features:**
- Interactive chat interface for GitHub repositories
- Support for multiple file types (Python, JavaScript, TypeScript, Markdown)
- Intelligent question validation using spaCy
- Real-time response generation
- Comprehensive error handling

**Tech Stack:**
- Frontend: Streamlit
- Vector Database: ChromaDB
- NLP: spaCy
- Document Processing: LlamaIndex

### 3. [AgentForge](./source_codes/agentforge/)
An AI-powered assistant that combines web search capabilities with creative image generation.

**Features:**
- Model Selection (OpenAI GPT-4 and HuggingFace Llama models)
- Web Search for events, conferences, and gatherings
- Image Generation from text descriptions
- File Upload support (PDF, DOCX, TXT)
- Gradio-based user interface

**Tech Stack:**
- Frontend: Gradio UI
- Framework: smolagents
- LLMs: GPT-4, Llama models
- File Processing: PDF, DOCX, TXT handlers
- Deployment: HuggingFace Spaces compatible

### 4. [Image Captioning](./source_codes/image-captioning/)
An image captioning system powered by Meta's Llama 3.2 Vision model, accessed through the Together AI platform.

**Features:**
- Advanced vision understanding capabilities
- Detailed image descriptions with technical focus
- Keyword extraction from images
- Support for multiple image formats
- Efficient streaming responses
- Error handling and validation

**Tech Stack:**
- Model: Llama-3.2-11B-Vision-Instruct-Turbo
- API Platform: Together AI
- Validation: Pydantic
- Environment Management: python-dotenv
- Base64 image processing

## Documentation
- [Multimodal-RAG Documentation](./source_codes/multimodal-rag/README.md)
- [Chat with GitHub Documentation](./source_codes/chat-with-gitrepo/readme.md)
- [AgentForge Documentation](./source_codes/agentforge/README.md)
- [Image Captioning Documentation](./source_codes/image-captioning/README.md)
