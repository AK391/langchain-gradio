import gradio as gr
from langchain_gradio import registry

registry(
    name='gpt-4-turbo',
    title='LangChain-Gradio Integration',
    description="Chat with gpt-4-turbo model.",
    examples=["Explain quantum gravity to a 5-year old.", "How many R are there in the word Strawberry?"]
).launch()
