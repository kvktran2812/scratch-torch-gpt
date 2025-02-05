import torch
import torch.nn as nn
from torch.nn import functional as F

def load_truyen_kieu_dataset():
    with open('data/truyen_kieu_clean.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print("Vocab size:", vocab_size)
    print("Number of characters:", len(text))

    return chars, text, vocab_size

def load_encoder_decoder(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    return encode, decode

def split_data(data, ratio: int = 0.9):
    n = int(ratio*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

def get_train_batch(data, batch_size:int = 64, block_size: int = 32):
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y