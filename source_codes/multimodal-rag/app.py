import streamlit as st
import requests
import hashlib
from pathlib import Path
import logging
import chromadb

# local imports
from utils.custom_styling import apply_custom_css

# --- Constants ---
TITLE = "Barista Bot: Your Coffee Crafting Companion ☕️"
SUBTITLE = "Guide to Perfecting Recipes!"
INPUT_PLACEHOLDER = "Ask anything about coffee:"
API_CONFIG = {
    "BARISTA_URL": "http://127.0.0.1:8000/api/v1/ask-barista-bot",
    "DATA_LOADER_URL": "http://127.0.0.1:8000/api/v1/load-data",
}
ADMIN_PASSWORD = "bd94dcda26fccb4e68d6a31f9b5aac0b571ae266d822620e901ef7ebe3a11d4f"
UPLOAD_PATH = "data"

# --- Initialization ---
def initialize_session_state():
    """Initialize or reset session state variables."""
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "doc_loaded" not in st.session_state:
        st.session_state.doc_loaded = False
    if "files_processed" not in st.session_state:
        st.session_state.files_processed = False

# --- Caching Decorators ---
@st.cache_resource
def get_chromadb_client():
    """Create a persistent ChromaDB client with caching."""
    return chromadb.PersistentClient(path="./chromadb")

@st.cache_data
def verify_password(input_password):
    """Verify admin password with caching."""
    hashed_input = hashlib.sha256(input_password.encode()).hexdigest()
    return hashed_input == ADMIN_PASSWORD

# --- Page Configuration ---
def setup_page_config():
    """Set up Streamlit page configuration."""
    st.set_page_config(
        page_title="Coffee Bot",
        page_icon="🤖",
    )

# --- Admin Features ---
def show_admin_features():
    """Display admin-specific features."""
    st.sidebar.success("Admin access granted!")
    try:
        # Initialize ChromaDB client and collection
        client = get_chromadb_client()
        collection = client.get_collection(name="multimodal_data_new")
        
        # Check if collection exists or is empty and display persistent messages
        if collection is None or collection.count() == 0:
            st.sidebar.warning("No collection/data found! Please upload documents to the database.")
                    
            # Button to load documents into DB
            if st.button("Load Documents to DB", key="load_db_button"):
                try:
                    res = requests.post(API_CONFIG["DATA_LOADER_URL"], timeout=600)
                    res.raise_for_status()
                    data = res.json()
                    if data["status"] == "success":
                        st.success(data["response"])
                        st.session_state.doc_loaded = True
                        st.sidebar.info("Documents loaded successfully!")
                    else:
                        st.error(data.get("response", "Error loading documents"))
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.sidebar.info("Collection found! You can upload new documents.")
            # File uploader and processing in the sidebar
            st.sidebar.subheader("Upload Custom Documents")
            uploaded_files = st.sidebar.file_uploader(
                "Choose files to add to the knowledge base",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'docx'],
                key="custom_document_uploader"
            )
            if uploaded_files:
                if st.sidebar.button("Process Uploaded Files", key="process_files_button"):
                    for file in uploaded_files:
                        file_path = Path(UPLOAD_PATH) / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    try:
                        res = requests.post(API_CONFIG["DATA_LOADER_URL"], timeout=600)
                        res.raise_for_status()
                        data = res.json()
                        if data["status"] == "success":
                            st.success(data["response"])
                            st.session_state.files_processed = True
                            st.sidebar.info("Uploaded files processed successfully!")
                            # Optionally display processing stats:
                            stats = data["data"]
                            st.write(f"Processed: {stats.get('processed', 0)} files")
                            if stats.get('failed', 0) > 0:
                                st.warning(f"Failed: {stats.get('failed', 0)} files")
                            if stats.get('unsupported', 0) > 0:
                                st.info(f"Unsupported: {stats.get('unsupported', 0)} files")
                        else:
                            st.error(data.get("response", "Unknown error occurred"))
                    except Exception as e:
                        st.error("Failed to process files. Please try again.")
                        logging.error(f"Error processing files: {str(e)}")
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")

# --- Chat Functions ---
def initialize_chat_history():
    """Initialize chat history in session state."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def generate_response(prompt, history):
    """
    Generate bot response and update conversation history.
    """
    history.append({"role": "user", "content": prompt})
    try:
        item = {"query": prompt}
        headers = {"Content-Type": "application/json"}
        with st.spinner("Processing your query..."):
            response = requests.post(
                API_CONFIG["BARISTA_URL"],
                json=item,
                headers=headers,
                timeout=600,
            )
            response.raise_for_status()
            response_data = response.json()
            if response_data["status"] == "success":
                bot_message = response_data["response"]
            else:
                bot_message = response_data.get("response", "Sorry, I couldn't process that.")
    except requests.exceptions.RequestException as e:
        bot_message = "I'm having trouble processing your request. Please try again later."
        logging.error(f"API request failed: {str(e)}")
    history.append({"role": "assistant", "content": bot_message})
    return bot_message, history

def display_chat_history(container):
    """Display all messages in the chat history."""
    with container:
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

def clear_input():
    """Clear the input field after submission."""
    st.session_state["user_input"] = st.session_state["widget"]
    st.session_state["widget"] = ""

# --- Main Application ---
def main():
    # Initialization
    initialize_session_state()
    setup_page_config()
    apply_custom_css()

    # User/Admin Selection in sidebar
    with st.sidebar:
        st.title("Access Level")
        user_type = st.radio("Select User Type:", ["User", "Admin"], index=0, key="user_role_selection")
        if user_type == "Admin":
            if not st.session_state.is_admin:
                password = st.text_input("Enter Admin Password:", type="password", key="admin_password")
                if password:
                    if verify_password(password):
                        st.session_state.is_admin = True
                    else:
                        st.error("Incorrect password!")
            if st.session_state.is_admin:
                show_admin_features()

    # Header
    st.image('static/images/Coffee_shop.jpg', use_container_width=True)
    st.title(TITLE)
    st.write(SUBTITLE)

    # Chat Interface
    initialize_chat_history()
    chat_container = st.container()
    
    # Text input for the chat interface with callback to clear input
    _user_input = st.text_input(
        INPUT_PLACEHOLDER,
        key="widget",
        on_change=clear_input,
        placeholder="Type your question here..."
    )
    
    # Process user query if there is input stored in session state
    if st.session_state.get("user_input"):
        _, st.session_state.conversation_history = generate_response(
            st.session_state["user_input"],
            st.session_state.conversation_history,
        )
        st.session_state["user_input"] = ""
    
    # Display chat history
    display_chat_history(chat_container)

if __name__ == "__main__":
    main()
