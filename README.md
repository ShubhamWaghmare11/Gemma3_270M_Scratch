# Gemma3 270M: Small Language Model Implementation from Scratch

This repository contains a **PyTorch implementation** of the **Google's Gemma3 model** with **270 million parameters**, trained from scratch on the **TinyStories** dataset.


## About

This model is based on Google DeepMind's Gemma3 architecture and was built from the ground up to explore training dynamics, architecture design, and generation quality of small LLMs. It includes advanced components such as:

- Sliding Window Attention (512-token window)

- Rotary Positional Embeddings (RoPE)

- RMSNorm for stable training

- Grouped Key-Value Attention (1 KV group)


## Model Architecture

- Parameters: 270M total (170M embedding + 100M transformer)
- Layers: 18 transformer blocks
- Attention Heads: 4 Query Heads, 1 KV Group
- Hidden Dimension: 2048
- Embedding Dimension: 640
- Head Dimension: 256
- Vocabulary Size: 50,257 (GPT-2 tokenizer)
- Context Length: 32,768 tokens (trained with 128 block size)
- Sliding Window: 512 tokens



## Training Details

- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) by Roneneldan

- Steps: 150,000 steps (not epochs)

- Batch Size: 32

- Loss Function: Cross-Entropy

- Optimizer: AdamW

- LR Scheduler: Linear Warmup + Cosine Decay

- Hardware: Single NVIDIA A100 GPU






## Requirements

```bash
pip install torch transformers tiktoken
```

## How to use

You can load and use the model with the Hugging Face `transformers` library:

```python
from transformers import AutoModelForCausalLM
import tiktoken
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Shubhamw11/Gemma3_270M_SLM_TinyStories")
tokenizer = tiktoken.get_encoding("gpt2")

#define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate text
input_text = "Once upon a time, there was a little"
context = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)
response = model.generate(context, max_new_tokens=200, temperature=1.1, top_k=5)
print(tokenizer.decode(response.squeeze().tolist()))


```

