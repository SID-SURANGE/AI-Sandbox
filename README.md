# AI-Sandbox
My personal AI sandbox for learning, experimenting, and prototyping different AI models and techniques.

## Projects

### 1. [(Multimodal RAG)](./source_codes/multimodal-rag/)
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

## Documentation
- [MM-RAG Documentation](./source_codes/MM-RAG/README.md)
- [Chat with GitHub Documentation](./source_codes/chat_with_gitrepo/readme.md)

## Prerequisites
- Python 3.7+
- Required Python packages listed in each project's requirements.txt
- API keys (GitHub token, OpenAI API key) as needed
