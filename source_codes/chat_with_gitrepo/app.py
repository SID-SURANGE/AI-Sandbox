# Standard library imports
import re
import subprocess
import time

# Third-party imports
import chromadb
import spacy
import streamlit as st

# LlamaIndex imports
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def parse_github_url(url):
    # Function Description -  This function parses a GitHub URL and returns the owner and repo name.
    
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)

    return match.groups() if match else (None, None)


def setup_chromadb(owner, repo):
    # Function Description -  This function sets up ChromaDB for storing embeddings.

    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(path="chromadb")
    chroma_collection = chroma_client.get_or_create_collection(f"{owner}_{repo}")

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context


def create_index(documents, storage_context):
    # Function Description - creates an index from the documents and returns a chat engine
    Settings.chunk_size = 1024
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    query_engine = index.as_query_engine()
    chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine)

    return chat_engine


def repo_data_extractor(github_client, owner, repo):
    #Function Description - This function fetches the data from the repository and returns it.
    
    # First validate if the repository exists
    try:
        # Try to get repository information
        repo_info = github_client.client.get_repo(f"{owner}/{repo}")

        # Check if repository is accessible and not empty
        if repo_info.private:
            return (
                None,
                "This is a private repository. Please provide access or use a public repository.",
            )
        if repo_info.fork:
            return (
                None,
                "This is a forked repository. Please use the original repository.",
            )
        if repo_info.archived:
            return None, "This repository is archived and may not be up to date."

    except Exception as e:
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg:
            return (
                None,
                f"Repository '{owner}/{repo}' not found. Please check if the owner and repository names are correct.",
            )
        elif "401" in error_msg or "403" in error_msg:
            return None, "Authentication failed. Please check your GitHub token."
        else:
            return None, f"Error accessing repository: {str(e)}"

    try:
        loader = GithubRepositoryReader(
            github_client,
            owner=owner,
            repo=repo,
            filter_file_extensions=(
                [".py", ".js", ".ts", ".md"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            verbose=False,
            concurrent_requests=5,
            use_parser=True,
        )
    except Exception as e:
        return None, f"Error initializing repository reader: {str(e)}"

    try:
        print(f"Loading {repo} repository by {owner}")
        docs = loader.load_data(branch="master")
        if not docs:
            return None, "No readable files found in the repository."
        print("Documents uploaded:")
        for doc in docs:
            print(doc.metadata)
        return docs, None
    except Exception as e:
        return None, f"Error loading repository data: {str(e)}"


def is_valid_question(text):
    # Function Description - This function checks if the user's input is a valid question.

    # Basic text cleanup
    text = text.strip()

    # Check if the text is too short
    if len(text) < 5:
        return False, "Question is too short. Please ask a proper question."

    # Check if it contains only random characters/numbers
    if not any(c.isalpha() for c in text):
        return False, "Question must contain actual words."

    # Process the text with spaCy
    doc = nlp(text.lower())

    # Check if it has at least 2 words
    if len([token for token in doc if not token.is_punct]) < 5:
        return False, "Please ask a complete question with at least 5 words."

    # Question detection using spaCy's features
    first_token = next(token for token in doc if not token.is_punct)

    # Define question indicators
    wh_deps = {"advmod", "nsubj", "attr"}  # Dependencies often found in questions
    aux_verbs = {"be", "do", "have", "can", "could", "would", "should", "will", "shall"}
    question_tags = {"WDT", "WP", "WP$", "WRB"}  # Question word POS tags

    is_question = (
        text.endswith("?")  # Ends with question mark
        or first_token.tag_ in question_tags  # Starts with WH-word
        or (
            first_token.dep_ in wh_deps and first_token.head.pos_ == "VERB"
        )  # Question structure
        or (first_token.lemma_ in aux_verbs)  # Starts with auxiliary verb
        or any(token.tag_ in question_tags for token in doc)  # Contains question words
    )

    if not is_question:
        return (
            False,
            "Please form a proper question. Start with question words (what, how, why), auxiliary verbs (is, are, do), or end with a question mark.",
        )

    return True, ""


def main():
    """Main function for the chat interface."""
    
    st.title("Chat with Git Repo")
    # GitHub token input (you might want to use a more secure method in production)
    github_token = st.text_input("Enter your GitHub token:", type="password")
    if not github_token:
        st.warning("Please enter your GitHub token.")
        return
    # GitHub URL input
    git_url = st.text_input("Enter GitHub repository URL:")
    if not git_url:
        st.warning("Please enter a GitHub repository URL.")
        return

    # Set up GitHub client
    github_client = GithubClient(github_token)

    # fetch the ownwer and repo name from the url
    owner, repo_name = parse_github_url(git_url)

    if owner and repo_name:
        if "chat_engine" not in st.session_state:
            # Create a placeholder for messages
            status_placeholder = st.empty()

            with status_placeholder:
                with st.spinner(f"Fetching data from {owner}/{repo_name}"):
                    data, error_msg = repo_data_extractor(
                        github_client, owner, repo_name
                    )
                    if data is None:
                        st.warning(error_msg)
                        return
                st.success(f"Data fetched from {owner}/{repo_name}")
                time.sleep(2)  # Show message for 2 seconds
            status_placeholder.empty()  # Clear the message

            status_placeholder = st.empty()
            with status_placeholder:
                with st.spinner("Setting up chromadb..."):
                    storage_context = setup_chromadb(owner, repo_name)
                st.success("Chromadb setup complete")
                time.sleep(2)
            status_placeholder.empty()

            status_placeholder = st.empty()
            with status_placeholder:
                with st.spinner("Creating index..."):
                    chat_engine = create_index(data, storage_context)
                st.success("Index created")
                time.sleep(2)
            status_placeholder.empty()

            st.session_state.chat_engine = chat_engine

        # Chat interface
        st.subheader("Chat with the Repository")
        user_question = st.text_input("Ask a question about the repository:")
        if user_question:
            is_valid, error_message = is_valid_question(user_question)
            if not is_valid:
                st.error(error_message)
            else:
                with st.spinner("Generating response..."):
                    response = st.session_state.chat_engine.chat(user_question)
                st.write("Response:", response.response)
    else:
        st.warning("Please enter a valid GitHub repository URL.")


if __name__ == "__main__":
    main()
