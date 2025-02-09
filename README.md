# truyen-kieu-gpt    
 
## Intro:
In this repo, I re-implement the GPT model, which is used in the famous ChatGPT. The model, however, is not the latest technologies of ChatGPT but rather the architecture from the early time like GPT-2 GPT-3. The model is decoder only taking from the Transformer model from "Attention is all you need" paper (Ashish Vaswani et al, 2017). Now, there are some details or features for the implementation:
- I trained the model with character-by-character tokenizer, not sub-word tokenizer so the model actually can't generate lots of meaningful sentence. It manage to generate Vietnamese tho.
- I have implemented the full pipeline in the train() function in script/gpt_utils.py file.
