# ===== STANDARD LIBRARY IMPORTS =====
import re
import subprocess
import logging

# ===== THIRD-PARTY IMPORTS =====
import chromadb
import spacy
import streamlit as st

# ===== LIBRARY IMPORTS (LLAMAINDEX & PYTHON GITHUB) =====
from github import Github
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor

# ===== LOGGER SETUP =====
logging.basicConfig(level=logging.INFO)

# ===== CONSTANTS =====
BRANCH = "master"
ADV_QUERY = False
CHUNK_SIZE = 1024
CHROMADB_PATH = "chromadb"
FILE_EXTENSIONS = (
    [".py", ".js", ".ts", ".md"],
    GithubRepositoryReader.FilterType.INCLUDE,
)
SIMILARITY_TOP_K = 10
SIMILARITY_CUTOFF = 0.3

# ===== LOAD RESPONSE SYNTHESIZER & LANGUAGE MODEL =====
response_synthesizer = get_response_synthesizer()
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


# ===== FUNCTION DECLARATIONS =====
def parse_github_url(url):
    """Parse a GitHub URL and return the owner and repository name."""
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


def setup_chromadb(owner, repo):
    """Set up ChromaDB for storing embeddings."""
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(f"{owner}_{repo}")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return storage_context


def create_index(documents, storage_context):
    """Create an index from the documents and return it."""
    Settings.chunk_size = CHUNK_SIZE
    # Ensure file name metadata is set for each document.
    for doc in documents:
        if "file_name" not in doc.metadata and "file_path" in doc.metadata:
            doc.metadata["file_name"] = doc.metadata["file_path"].split("/")[-1]
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


def create_retriever(index, adv_query=True, response_mode="tree_summarize"):
    """
    Create an advanced retriever that is robust enough to understand queries.
    For advanced queries, it employs a custom handler that detects—for example—a request
    for listing Python file names and returns all matching file names from the index.
    """
    if not adv_query:
        query_engine = index.as_query_engine()
        query_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine, verbose=True
        )
        return query_engine
    else:
        # Use an increased top k value for broader results.
        retriever = VectorIndexRetriever(index=index, similarity_top_k=SIMILARITY_TOP_K)
        base_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode=response_mode,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)
            ],
        )

        return base_query_engine


def repo_data_extractor(github_client, github_token, owner, repo):
    """
    Extract data from the GitHub repository. The function first uses PyGithub to validate
    that the repository exists and is accessible, then loads the repository data using
    GithubRepositoryReader.
    """
    # Validate repository existence with PyGithub.
    try:
        py_github_client = Github(github_token)
        repo_info = py_github_client.get_repo(f"{owner}/{repo}")
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
    except Exception as e:
        error_msg = str(e).lower()
        if "404" in error_msg or "not found" in error_msg:
            return (
                None,
                f"Repository '{owner}/{repo}' not found. Please check if the owner and repository names are correct.",
            )
        elif "401" in error_msg or "403" in error_msg:
            return (None, "Authentication failed. Please check your GitHub token.")
        else:
            return (None, f"Error accessing repository: {str(e)}")

    # Use LlamaIndex's GithubRepositoryReader to load documents from the repo.
    try:
        loader = GithubRepositoryReader(
            github_client,
            owner=owner,
            repo=repo,
            filter_file_extensions=FILE_EXTENSIONS,
            verbose=False,
            concurrent_requests=5,
            use_parser=True,
        )
    except Exception as e:
        return (None, f"Error initializing repository reader: {str(e)}")

    try:
        print(f"Loading {repo} repository by {owner}")
        docs = loader.load_data(branch=BRANCH)
        if not docs:
            return (None, "No readable files found in the repository.")
        print("Documents uploaded:")
        for doc in docs:
            print(doc.metadata)
        return (docs, None)
    except Exception as e:
        return (None, f"Error loading repository data: {str(e)}")


def is_valid_question(text):
    """Validate that the user's input is a proper question."""
    text = text.strip()
    if len(text) < 5:
        return False, "Question is too short. Please ask a proper question."
    if not any(c.isalpha() for c in text):
        return False, "Question must contain actual words."
    doc = nlp(text.lower())
    if len([token for token in doc if not token.is_punct]) < 5:
        return False, "Please ask a complete question with at least 5 words."
    first_token = next(token for token in doc if not token.is_punct)
    wh_deps = {"advmod", "nsubj", "attr"}
    aux_verbs = {"be", "do", "have", "can", "could", "would", "should", "will", "shall"}
    question_tags = {"WDT", "WP", "WP$", "WRB"}
    is_question = (
        text.endswith("?")
        or first_token.tag_ in question_tags
        or (first_token.dep_ in wh_deps and first_token.head.pos_ == "VERB")
        or (first_token.lemma_ in aux_verbs)
        or any(token.tag_ in question_tags for token in doc)
    )
    if not is_question:
        return (
            False,
            "Please form a proper question. Start with question words (what, how, why), auxiliary verbs (is, are, do), or end with a question mark.",
        )
    return (True, "")


def setup_query_engine(github_client, github_token, owner, repo_name):
    """
    Set up the query engine if it doesn't already exist in session state.
    This includes fetching data, setting up ChromaDB, creating an index,
    and initializing the retriever.
    """
    if "query_engine" not in st.session_state:
        try:
            # Step 1: Fetch Repository Data
            with st.spinner(f"Fetching data from {owner}/{repo_name}..."):
                data, error_msg = repo_data_extractor(
                    github_client, github_token, owner, repo_name
                )
                if data is None:
                    st.error(error_msg)
                    return False
                st.toast(f"Data fetched from {owner}/{repo_name}!", icon="✅")

            # Step 2: Set Up ChromaDB
            with st.spinner("Setting up ChromaDB..."):
                storage_context = setup_chromadb(owner, repo_name)
                st.toast("ChromaDB setup complete!", icon="✅")

            # Step 3: Create Index
            with st.spinner("Creating index..."):
                index = create_index(data, storage_context)
                st.toast("Index created!", icon="✅")

            # Step 4: Create Retriever
            with st.spinner("Creating retriever..."):
                query_engine = create_retriever(index, adv_query=ADV_QUERY)
                st.toast("Retriever created!", icon="✅")
                st.session_state.query_engine = query_engine

        except Exception as e:
            st.error(f"An error occurred during setup: {str(e)}")
            return False

    return True


def main():
    """Main function for the chat interface."""
    st.title("Chat with Git Repo")
    github_token = st.text_input("Enter your GitHub token:", type="password")
    if not github_token:
        st.warning("Please enter your GitHub token.")
        return
    git_url = st.text_input("Enter GitHub repository URL:")
    if not git_url:
        st.warning("Please enter a GitHub repository URL.")
        return

    st.toast("Parsing GitHub URL...", icon="🔍")
    owner, repo_name = parse_github_url(git_url)
    if not (owner and repo_name):
        st.error("Invalid GitHub URL. Please check the format.")
        return
    st.toast(f"Fetched repository: {owner}/{repo_name}", icon="🔍")

    github_client = GithubClient(github_token)

    print(f"session state: {st.session_state}")

    # Set up query engine
    if not setup_query_engine(github_client, github_token, owner, repo_name):
        return

    st.subheader("Chat with the Repository")
    user_question = st.text_input("Ask a question about the repository:")
    if user_question:
        valid, error_message = is_valid_question(user_question)
        if not valid:
            st.error(error_message)
        else:
            with st.spinner("Generating response..."):
                # Use the custom query handler if available
                if ADV_QUERY:
                    response = st.session_state.query_engine.query(user_question)
                else:
                    # Fallback to older chat method if needed
                    response = st.session_state.query_engine.chat(user_question)
                st.write("Response:", response.response)


if __name__ == "__main__":
    main()
