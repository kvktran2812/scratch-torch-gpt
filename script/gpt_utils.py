import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

def load_truyen_kieu_dataset(text_file: str = 'data/truyen_kieu_clean.txt'):
    with open(text_file, 'r', encoding='utf-8') as f:
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

def get_batch(data, batch_size:int = 64, block_size: int = 32):
    idx = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x, y

def beautiful_print(model, decoder, max_new_tokens, block_size: int = 32, temperature=1.0, top_k=None, device='cpu'):
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    poem = model.generate(context, max_new_tokens, block_size, temperature, top_k)
    poem = decoder(poem[0].tolist())

    for i in range(1, len(poem)):
        print(poem[i], end='')
        time.sleep(0.05)

    return poem

@torch.no_grad()
def eval(model, val_data, device: str = 'cpu', n_iters: int = 100):
    model.eval()
    losses = torch.zeros(n_iters)
    for k in range(n_iters):
        x_val, y_val = get_batch(val_data)
        x_val, y_val = x_val.to(device), y_val.to(device)
        logits, loss = model(x_val, y_val)
        losses[k] = loss.item()
    model.train()
    return loss.mean()

def train(
    model, 
    optimizer, 
    device, 
    train_data,
    val_data,
    n_steps: int = 1000, 
    train_iter: int = 100, 
    eval_iter: int = 100, 
    total_eval: int = 100, 
    save_checkpoint: int = 100, 
    save_dir: str = "checkpoints"
):
    # Preparation
    model.to(device)
    loss_history = []
    save = 1

    # Check if save dir valid
    if not os.path.exists(save_dir):
        print("Directory not exist, creating directory...")
        os.makedirs(save_dir)
        print(f"Directory {save_dir} created")
    else:
        print(f"Directory {save_dir} exists")
    print("Start training...")
    
    for step  in range(n_steps):
        model.train()
        x_train, y_train = get_batch(train_data)
        x_train, y_train = x_train.to(device), y_train.to(device)
        logits, loss = model(x_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        # Evaluation
        if step != 0:
            if step % train_iter == 0:
                avg_loss = sum(loss_history[-train_iter:]) / train_iter  # Compute average over last eval_iter steps
                print(f"Step {step}: Average Loss = {avg_loss:.4f}")
        
            if step % eval_iter == 0:
                eval_loss = eval(model, val_data, device, total_eval)
                print(f"***Eval: {eval_loss:.4f}")
    
            if step % save_checkpoint == 0:
                torch.save(model.state_dict(), f"{save_dir}/checkpoint_{int(save)}.pth")
                save += 1

    # Final eval:
    eval_loss = eval(model, val_data, device, total_eval)
    print("*" * 10)
    avg_loss = sum(loss_history) / n_steps
    print(f"Final eval --- Step: {n_steps} - Training Loss: {avg_loss:.4f} - Eval Loss: {eval_loss:.4f}")
    torch.save(model.state_dict(), f"{save_dir}/checkpoint_final.pth")

    return loss_history, eval_loss