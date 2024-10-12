import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr
from typing import Callable, Dict, Any

__version__ = "0.0.1"


def get_chat_model(model_name: str, api_key: str | None = None) -> BaseChatModel:
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model_name=model_name, openai_api_key=api_key, streaming=True)
    elif model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name, anthropic_api_key=api_key, streaming=True)
    elif model_name.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, streaming=True)
    elif "/" in model_name:
        if api_key:
            # Use HuggingFaceEndpoint for models that require API access
            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                max_new_tokens=100,
                do_sample=False,
                huggingfacehub_api_token=api_key
            )
            return ChatHuggingFace(llm=llm)
        else:
            # Use HuggingFacePipeline for local models
            return HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 100,
                    "top_k": 50,
                    "temperature": 0.7,
                },
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    chat = get_chat_model(model_name, api_key)
    memory = ConversationBufferMemory(return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    def fn(message, history):
        inputs = preprocess(message, history)
        response = conversation.predict(input=inputs["user_message"])
        yield postprocess(response)

    return fn


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            return {"user_message": message}

        postprocess = lambda x: x  # No post-processing needed
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name):
    # For now, all supported models are chat models
    return "chat"


def registry(name: str, token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a supported LangChain chat model.

    Parameters:
        - name (str): The name of the model (e.g., "gpt-3.5-turbo", "gpt-4", "claude-2", "gemini-pro", "microsoft/phi-2").
        - token (str, optional): The API key for the model provider. For Hugging Face models, this is optional and only needed for accessing gated or private models.
    """
    if name.startswith("gpt-"):
        env_var_name = "OPENAI_API_KEY"
    elif name.startswith("claude-"):
        env_var_name = "ANTHROPIC_API_KEY"
    elif name.startswith("gemini-"):
        env_var_name = "GOOGLE_API_KEY"
    elif "/" in name:
        env_var_name = "HUGGINGFACEHUB_API_TOKEN"
    else:
        raise ValueError(f"Unsupported model: {name}")

    api_key = None
    if env_var_name:
        api_key = token or os.environ.get(env_var_name)
        if not api_key and "/" not in name:
            raise ValueError(
                f"API key for {name} is not set. "
                f"Please set the {env_var_name} environment variable "
                f"or provide the token parameter."
            )

    pipeline = get_pipeline(name)
    inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
    fn = get_fn(name, preprocess, postprocess, api_key)

    if pipeline == "chat":
        interface = gr.ChatInterface(
            fn=fn,
            chatbot=gr.Chatbot(height=600),
            textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
            title=f"Chat with {name}",
            description="This is a chatbot powered by LangChain and various AI models.",
            theme="soft",
            examples=["Tell me a joke", "Explain quantum computing", "What's the weather like today?"],
            cache_examples=True,
            **kwargs
        )
    else:
        # For other pipelines, create a standard Interface (not implemented yet)
        interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface
