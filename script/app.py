import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from gpt_utils import *
import gradio as gr
from gpt import GPT

model = GPT(512, 129, 512, 32, 6)
model.load_state_dict(torch.load("../checkpoints/checkpoint_final.pth", weights_only=True))
model.eval()

chars, text, vocab_size = load_truyen_kieu_dataset("../data/truyen_kieu_clean.txt")
encoder, decoder = load_encoder_decoder(chars)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate():
    poem = beautiful_print(model, decoder, 400, device=device)
    return poem

iface = gr.Interface(
    fn=generate, 
    inputs=None, 
    outputs="text",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()