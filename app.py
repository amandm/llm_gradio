import gradio as gr
import random
import time

from gpt4all import GPT4All
import io

gpt = GPT4All("ggml-model-gpt4all-falcon-q4_0")


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history

def generator(question):
    output = gpt.generate(question, max_tokens=200,streaming = True)
    for token in output:
        pass
    return "", token


def bot(history):
    question = history[-1][0]
    response = gpt.generate(question, max_tokens=200,streaming = True)
    # print("response",response)
    history[-1][1] = ""
    # print(history)
    for character in response:
        # print(character)
        history[-1][1] += character
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload an image",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch()