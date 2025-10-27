import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline


text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6",
                dtype=torch.bfloat16)

# model_path =("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots"\
#             "/a4f8f3ea906ed274767e9906dbaede7531d660ff")
# text_summary = pipeline("summarization", model=model_path,
#                 dtype= torch.bfloat16)

# text="Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for
# his leadership of Tesla, SpaceX, Twitter, and xAI. Musk has been the wealthiest person in the world since
# 2021; as of October 2025, Forbes estimates his net worth to be US$500 billion."
# print(text_summary(text));


def summarize(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()
# demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
demo = gr.Interface(fn=summarize,
                    inputs=[gr.Textbox(label="Input text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text", lines=4)],
                    title = "Project 1:Text Summarizer",
                    description="This application will be used to summarize the text")
demo.launch()