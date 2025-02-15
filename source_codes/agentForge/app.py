# System Imports
import os
import yaml

# Third-party Imports
from dotenv import load_dotenv

# Local Imports
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool, LiteLLMModel
from smolagents.monitoring import LogLevel
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

# Load environment variables
load_dotenv()

def verify_api_keys():
    """Verify that required API keys are present and valid."""
    openai_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    if not hf_token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
    
    # Basic format verification
    if not openai_key.startswith("sk-"):
        raise ValueError("Invalid OPENAI_API_KEY format")
    if not hf_token.startswith("hf_"):
        raise ValueError("Invalid HUGGING_FACE_HUB_TOKEN format")
    
    return True

# Verify API keys before proceeding
verify_api_keys()

@tool
def generate_image(prompt: str) -> str:
    """
    Important! - This tool is the ONLY tool that should be used to generate images.
    Use this tool exclusively for image generation tasks.
    Args:
        prompt: The prompt to generate the image from.
    Returns:
        str: The generated image.
    """
    image_generation_tool = load_tool(
        "agents-course/text-to-image", trust_remote_code=True
    )
    return image_generation_tool(prompt)


@tool
def enhanced_web_search(query: str) -> str:
    """
    A wrapper tool for the DuckDuckGo search tool with enhanced functionality.

    Args:
        query: The search query to perform.

    Returns:
        str: Formatted search results including titles, URLs, and snippets.

    Description:
        This tool performs a web search using DuckDuckGo and returns the top search results
        in a formatted string. It's useful for quickly gathering information from the internet.
    """
    try:
        search_tool = DuckDuckGoSearchTool(max_results=5)
        results = search_tool(query)

        # If we get results in the expected format
        if "\n\n" in results:
            formatted_results = "Search Results:\n\n"
            for i, result in enumerate(results.split("\n\n")[1:], 1):
                formatted_results += f"{i}. {result}\n\n"
            return formatted_results.strip()

        # If we get results in a different format, return as is
        return f"Search Results:\n\n{results}"

    except Exception as e:
        # If the search fails, try with a simpler query
        try:
            # Remove any numbers and simplify the query
            simplified_query = query.replace("top 3", "").replace("top 5", "").strip()
            search_tool = DuckDuckGoSearchTool(max_results=3)
            results = search_tool(simplified_query)
            return f"Search Results (simplified query):\n\n{results}"
        except Exception as e2:
            return f"Search failed. Please try a different query. Error: {str(e2)}"


final_answer = FinalAnswerTool()

def create_model(model_choice="OpenAI GPT-4"):
    if model_choice == "HuggingFace (Llama-3.2-1B)":
        return HfApiModel(
            token=os.getenv("HUGGING_FACE_HUB_TOKEN"),
            max_tokens=1000,
            temperature=0.4,
            model_id="meta-llama/Llama-3.2-1B-Instruct",
            custom_role_conversions=None,
        )
    else:  # OpenAI GPT-4
        return LiteLLMModel(
            model_id="openai/gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

# Initialize with default model (OpenAI GPT-4)
model = create_model()

with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        enhanced_web_search,
        generate_image,
        final_answer,
    ],
    max_steps=6,
    verbosity_level=LogLevel.DEBUG,
    grammar=None,
    planning_interval=None,
    name="AgentForge",
    description="An intelligent agent that combines web search and creative capabilities to help solve complex tasks.",
    prompt_templates=prompt_templates,
    additional_authorized_imports=["json", "datetime", "re", "matplotlib"]
)

# Initialize the Gradio UI with our agent
GradioUI(agent).launch()
