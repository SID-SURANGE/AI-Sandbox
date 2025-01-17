import streamlit as st
import requests
import hashlib
from pathlib import Path

# local imports
from utils.custom_styling import apply_custom_css

# --- Constants ---
TITLE = "Barista Bot 🤖: Your Coffee Crafting Companion ☕️"
SUBTITLE = "Guide to Perfecting Recipes!"


INPUT_PLACEHOLDER = "Ask anything about coffee:"
API_CONFIG = {
    "BARISTA_URL": "http://127.0.0.1:8000/api/v1/ask-barista-bot",  # Replace with your actual API endpoint
    "DATA_LOADER_URL": "http://127.0.0.1:8000/api/v1/load-data",
}

# Add these constants
ADMIN_PASSWORD = "bd94dcda26fccb4e68d6a31f9b5aac0b571ae266d822620e901ef7ebe3a11d4f"  # Store hashed password, not plaintext
UPLOAD_PATH = "data"

# --- Styling ---
def setup_page_config():
    """Set up Streamlit page configuration."""
    st.set_page_config(
        page_title="Coffee Bot",
        page_icon="🤖",
    )


def verify_password(input_password):
    """Verify admin password."""
    hashed_input = hashlib.sha256(input_password.encode()).hexdigest()
    is_valid = hashed_input == ADMIN_PASSWORD
    if is_valid:
        st.session_state.is_admin = True
    return is_valid

def show_admin_features():
    """Display admin-specific features."""
    st.sidebar.success("Admin access granted!")
    
    with st.sidebar:
        st.subheader("Upload Custom Documents")
        uploaded_files = st.file_uploader(
            "Choose files to add to the knowledge base",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'doc', 'docx', 'jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi']
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files"):
                for file in uploaded_files:
                    # Save file
                    file_path = Path(UPLOAD_PATH) / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Add to vector DB (you'll need to implement this)
                    # process_and_store_document(file_path)
                    res = requests.post(API_CONFIG["DATA_LOADER_URL"], timeout=600)
                    if res.status_code != 200:
                        st.error(f"Failed to add knowledge: {res.text}")
                    else:
                        st.success("Files processed and added to knowledge base!")
                    res.raise_for_status()

# --- Chat Functions ---
def initialize_chat_history():
    """Initialize chat history in session state."""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def generate_response(prompt, history):
    """
    Generate bot response and update conversation history.
    
    Args:
        prompt (str): User input message.
        history (list): Conversation history.
    
    Returns:
        tuple: (bot_message, updated_history)
    """
    history.append({"role": "user", "content": prompt})
    
    # Simulate API call for bot response
    try:
        item = {"query": prompt}
        headers = {"Content-Type": "application/json"}
        
        # Add a loader while waiting for a response
        with st.spinner("Processing your query..."):
            response = requests.post(
                API_CONFIG["BARISTA_URL"],
                json=item,
                headers=headers,
                timeout=600,
            )
        
        response.raise_for_status()
        bot_message = response.json().get("response", "Sorry, I couldn't process that.")
    
    except requests.exceptions.RequestException as e:
        bot_message = f"Error fetching response: {e}"
    
    history.append({"role": "assistant", "content": bot_message})
    return bot_message, history

def display_chat_history(container):
    """
    Display all messages in the chat history.
    
    Args:
        container: Streamlit container object.
    """
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
    # Setup
    setup_page_config()
    apply_custom_css()

    # User/Admin Selection
    with st.sidebar:
        st.title("Access Level")
        user_type = st.radio("Select User Type:", ["User", "Admin"], index=0)
        
        if user_type == "Admin":
            password = st.text_input("Enter Admin Password:", type="password")
            if password:
                if verify_password(password):
                    show_admin_features()
                else:
                    st.error("Incorrect password!")
    
    # Header
    st.image('static\images\Coffee_shop.jpg', use_container_width=True)
    st.title(TITLE)
    st.write(SUBTITLE)
    
    # Initialize chat
    initialize_chat_history()
    
    # Create chat container
    chat_container = st.container()
    
    # Text input with a callback that fires on 'Enter'
    # Store user input in session state to persist between reruns
    _user_input = st.text_input(
        INPUT_PLACEHOLDER,
        key="widget",
        on_change=clear_input,
        placeholder="Type your question here...",
    )
    
    # If there is a newly submitted user query, generate a response
    if st.session_state.get("user_input"):
        _ , st.session_state.conversation_history = generate_response(
            st.session_state["user_input"],
            st.session_state.conversation_history,
        )
        
        # Optionally clear user_input after processing
        st.session_state["user_input"] = ""
    
    # Display chat
    display_chat_history(chat_container)

if __name__ == "__main__":
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    main()
