import gradio as gr
from transformers import pipeline

pipe = pipeline("sentiment-analysis")


def greet(text: str):
    return pipe(text)


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="json",
    examples=[
        "I love this",
        "I hate this",
    ],
)

demo.launch(server_port=8080)
