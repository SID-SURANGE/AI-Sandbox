 # GitHub Repository Chat

A Streamlit-based application that allows users to chat with and ask questions about any public GitHub repository. The application uses LlamaIndex for document processing and spaCy for natural language processing.

## Features

- Chat interface for GitHub repositories
- Support for multiple file types (Python, JavaScript, TypeScript, Markdown)
- Intelligent question validation using spaCy
- Real-time response generation
- Progress indicators and status messages
- Comprehensive error handling for repository access

## Prerequisites

- Python 3.7+
- GitHub Personal Access Token
- Public GitHub repository URL

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Environment Setup

Create a `.env` file in the project root and add your GitHub token:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Enter your GitHub token when prompted
3. Provide a public GitHub repository URL
4. Ask questions about the repository

## Features in Detail

### Repository Processing
- Supports `.py`, `.js`, `.ts`, and `.md` files
- Validates repository accessibility
- Checks for private, forked, or archived repositories
- Concurrent file processing for better performance

### Question Validation
- Minimum 5-word requirement
- Natural language processing using spaCy
- Detects various question types:
  - WH-questions (what, why, how, etc.)
  - Yes/No questions
  - Questions with auxiliary verbs
  - Questions ending with '?'

### Error Handling
- Repository not found
- Authentication failures
- Private repository access
- Invalid repository URLs
- Data loading issues

## Limitations

- Only works with public repositories (unless you have access to private ones)
- Limited to specific file types (.py, .js, .ts, .md)
- Requires valid GitHub token with appropriate permissions
- Repository size may affect processing time

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)