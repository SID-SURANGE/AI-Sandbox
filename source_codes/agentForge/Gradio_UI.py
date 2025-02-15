#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from tabnanny import verbose
from typing import Optional
import gradio as gr
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available
from smolagents import CodeAgent
from smolagents.monitoring import LogLevel

def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        from app import create_model  # Import the model creation function
        self.create_model = create_model
        self.base_agent = agent
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def verify_api_keys(self):
        """Verify if API keys are present in .env file"""
        openai_key = os.getenv("OPENAI_API_KEY")
        hf_key = os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        missing_keys = []
        if not openai_key:
            missing_keys.append("OpenAI API Key")
        if not hf_key:
            missing_keys.append("HuggingFace Hub Token")
            
        return len(missing_keys) == 0, missing_keys

    def update_model_dropdown_state(self, api_verified):
        """Update model dropdown state based on API verification"""
        keys_present, missing_keys = self.verify_api_keys()
        
        if not api_verified or not keys_present:
            if missing_keys:
                message = f"⚠️ Missing required API keys: {', '.join(missing_keys)}"
            else:
                message = "⚠️ Please verify your API keys in .env file first"
            return (
                gr.update(value="OpenAI GPT-4", interactive=False), 
                message,
                gr.update(value=False)  # Uncheck the checkbox if keys are missing
            )
        return (
            gr.update(interactive=True), 
            "✅ API keys verified. You can now select your preferred model",
            gr.update(value=True)
        )

    def update_model(self, model_choice):
        new_model = self.create_model(model_choice)
        # Get the actual Tool objects from the base agent's tools dictionary
        tools = list(self.base_agent.tools.values())
        self.agent = CodeAgent(
            model=new_model,
            tools=tools,
            max_steps=6,  # Match app.py
            verbosity_level=LogLevel.DEBUG,  # Match app.py's DEBUG level
            grammar=self.base_agent.grammar,
            planning_interval=self.base_agent.planning_interval,
            name=self.base_agent.name,
            description=self.base_agent.description,
            prompt_templates=self.base_agent.prompt_templates,
            additional_authorized_imports=self.base_agent.additional_authorized_imports
        )
        return f"🔄 Currently using: **{model_choice}**"

    def interact_with_agent(self, prompt, messages):
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(
            title="AgentForge",
            css="""
                .container {
                    text-align: center;
                    margin-bottom: 2rem;
                }
                .main-title {
                    font-size: 3.5em;
                    font-weight: bold;
                    color: #2B65EC;
                    margin-bottom: 0.2em;
                }
                .subtitle {
                    font-size: 1.2em;
                    color: #5D76A9;
                    font-style: italic;
                    margin-top: 0.5em;
                }
                .app-description {
                    margin: 1em 0;
                    text-align: left;
                }
            """
        ) as demo:
            with gr.Column():
                gr.HTML(
                    """
                    <div class="container">
                        <div class="main-title">AgentForge</div>
                        <div class="subtitle">Crafting Intelligence, Forging Solutions</div>
                    </div>
                    """
                )
            
            gr.Markdown(
                """
                <div class="app-description">
                Welcome to AgentForge - your intelligent assistant powered by AI! This app combines powerful web search capabilities 
                with creative image generation. You can use it to:
                
                - Search for events, conferences, and gatherings worldwide
                - Generate creative images from text descriptions
                - Get detailed answers with web-based research
                </div>
                """
            )

            with gr.Accordion("⚠️ Important: Configure Model & API Keys First", open=False):
                with gr.Column():
                    gr.Markdown("""
                    <br/>

                    ### 🔑 API Key Requirements <i>(to be taken care when the space is cloned)</i>
                    Before using the models, please ensure you have:
                    1. **OpenAI API Key** - Required for GPT-4 model
                    2. **HuggingFace Hub Token** - Required for Llama model
                    
                    Add these to your `.env` file as:
                    ```
                    OPENAI_API_KEY=your_key_here
                    HUGGING_FACE_HUB_TOKEN=your_token_here
                    ```
                    """)
                    api_verified = gr.Checkbox(
                        label="I have added the required API keys to .env file",
                        value=False,  # Always start unchecked
                        info="Check this box only after adding your API keys"
                    )
                
                    model_dropdown = gr.Dropdown(
                        choices=["OpenAI GPT-4", "HuggingFace (Llama-3.2-1B)"],
                        value="OpenAI GPT-4",
                        label="Select AI Model",
                        container=True,
                        scale=1,
                        min_width=200,
                        info="Choose the AI model to power your agent",
                        interactive=False  # Start disabled
                    )
                    model_status = gr.Markdown(
                        visible=True,
                        value="⚠️ Please verify your API keys in .env file first"
                    )
            
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png",
                ),
                scale=1,
            )
            
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )

            text_input = gr.Textbox(lines=1, label="Chat Message")

            # Add API verification handler
            api_verified.change(
                self.update_model_dropdown_state,
                inputs=[api_verified],
                outputs=[model_dropdown, model_status, api_verified]
            )

            # Add model selection handler
            model_dropdown.change(
                self.update_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )

            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot],
                [chatbot]
            )

        demo.launch(debug=True, share=True, **kwargs)


__all__ = ["stream_to_gradio", "GradioUI"]