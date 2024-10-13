import gradio as gr
import langchain_gradio

with gr.Blocks() as demo:
    with gr.Tab("GPT-3.5-turbo"):
        gr.load('gpt-3.5-turbo', src=langchain_gradio.registry)
    with gr.Tab("Claude-3"):
        gr.load('claude-3-5-sonnet-20240620', src=langchain_gradio.registry)
    with gr.Tab("gemini-1.5-pro-002"):
        gr.load('gemini-1.5-pro-002', src=langchain_gradio.registry)
    with gr.Tab("Hugging Face"):
        gr.load('Qwen/Qwen2.5-72B-Instruct', src=langchain_gradio.registry)

demo.launch()