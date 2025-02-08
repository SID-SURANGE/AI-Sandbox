
import streamlit as st

def apply_custom_css():
    """Apply custom CSS for styling."""
    st.markdown(
        """
        <style>
        /* Main app background */
        .stApp {
            background-color: #F5F5F5; /* Light gray background */
        }

        /* Main chat input box styling */
        .stTextInput > div > div > input {
            background-color: white !important;
            color: black !important;
            border-radius: 5px;
            border: 1px solid #ccc;
        }

        /* Chat message bubbles styling */
        .stChatMessage {
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 8px;
        }

        /* Sidebar container styling */
        [data-testid="stSidebar"] {
            background-color: #2E4374;  /* Dark blue background for sidebar */
            padding: 1rem;
        }

        /* All text elements in sidebar */
        [data-testid="stSidebar"] * {
            color: white !important;
        }

        /* Radio button labels in sidebar */
        [data-testid="stSidebar"] .stRadio label {
            color: white !important;
        }

        /* Text input fields in sidebar */
        [data-testid="stSidebar"] input[type="text"] {
            color: black !important;
        }

        /* File upload status message */
        [data-testid="stSidebar"] .stUploadedFileMsg {
            color: white !important;
        }
        
        /* File uploader component text */
        [data-testid="stFileUploader"] {
            color: white !important;
        }
        
        /* File uploader button */
        [data-testid="stFileUploader"] button {
            color: black !important;
            background-color: #E5E5E5 !important;
        }
        
        /* Uploaded file name text */
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
            color: white !important;
        }
        
        /* File upload status indicator */
        .uploadedFileName {
            color: white !important;
        }
        
        /* All buttons in sidebar (including 'Process Uploaded Files') */
        [data-testid="stSidebar"] button {
            color: black !important;
            background-color: #E5E5E5 !important;
            border: 1px solid #E5E5E5 !important;
        }

        /* Specifically target button text */
        [data-testid="stSidebar"] button p {
            color: black !important;
        }

        /* Target button inner elements */
        [data-testid="stSidebar"] button div {
            color: black !important;
        }

        /* Target button span elements */
        [data-testid="stSidebar"] button span {
            color: black !important;
        }
        
        /* Password input field */
        [data-testid="stSidebar"] input[type="password"] {
            color: black !important;
        }
        
        /* Password visibility toggle icon */
        [data-testid="stSidebar"] .stPasswordInput > button {
            color: black !important;
            background: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
