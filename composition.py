import gradio as gr
from langchain_gradio import registry

with gr.Blocks() as demo:
    with gr.Tab("GPT-4-turbo"):
        registry("gpt-4-turbo")
    with gr.Tab("GPT-3.5-turbo"):
        registry("gpt-3.5-turbo")

demo.launch()
