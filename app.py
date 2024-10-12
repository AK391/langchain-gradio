import gradio as gr
import langchain_gradio

gr.load(
    name='gpt-4-turbo',
    src=langchain_gradio.registry,
).launch()
