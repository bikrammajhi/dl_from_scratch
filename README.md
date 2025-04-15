# 🦙 LLaMA 2 from Scratch

<div align="center">
  <img src="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/transformer_llama.png" alt="LLaMA 2 Banner" width="700px">

  <p><em>A comprehensive implementation of Meta AI's LLaMA 2 architecture with detailed explanations and visualizations</em></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
</div>

## 📋 Table of Contents

- [Model Overview](#-model-overview)
- [Architecture Details](#-architecture-details)
- [Implementation Highlights](#-implementation-highlights)
  - [KV-Caching for Efficient Inference](#kv-caching-for-efficient-inference)
  - [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
  - [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
  - [RMSNorm & SwiGLU Activation](#rmsnorm--swiglu-activation)
- [Code Structure](#-code-structure)
- [Getting Started](#-getting-started)

---

## 🔍 Model Overview

LLaMA 2 is a state-of-the-art large language model that builds upon the foundation of the original LLaMA while introducing significant improvements:

- **Increased Context Length**: From 2048 to 4096 tokens
- **Enhanced Training Dataset**: 40% more data than the original LLaMA
- **Optimized Inference Performance**: Through KV-caching and GQA
- **Improved Instruction Following**: Fine-tuned with RLHF (Reinforcement Learning from Human Feedback)

The model comes in various sizes (7B, 13B, and 70B parameters), making it adaptable to different computational constraints while maintaining impressive capabilities.

---

## 🏗 Architecture Details

<div align="center">
  <img src="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/transformer_vs_llama.svg" alt="LLaMA 2 Detailed Architecture" width="600px">
  <p><em>Detailed view of LLaMA 2's transformer-based architecture</em></p>
</div>


LLaMA 2 follows a decoder-only transformer architecture, similar to GPT models, with several key optimizations:

| Component | Description |
|-----------|-------------|
| Decoder Blocks | Multiple transformer layers with self-attention and feed-forward networks |
| Pre-normalization | RMSNorm applied before each sub-layer for stable training |
| SwiGLU Activation | Enhanced activation function in the feed-forward network |
| Rotary Embeddings | Position-aware token representations |
| Grouped-Query Attention | Efficient attention mechanism that reduces computational load |

---

## 💡 Implementation Highlights

### KV-Caching for Efficient Inference

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*WAJVBEY5vs6QHh2W.gif" alt="KV-Caching Diagram" width="650px">
  <p><em>KV caching mechanism used to accelerate autoregressive token generation</em></p>
</div>

KV-Caching is a critical optimization technique that dramatically speeds up inference during autoregressive text generation:

```python
# Pseudocode for KV-caching implementation
def forward_with_kv_cache(self, x, kv_cache=None):
    if kv_cache is None:
        # First forward pass - compute and cache all keys and values
        keys, values = self.compute_kv(x)
        kv_cache = (keys, values)
        return self.attention(x, keys, values), kv_cache
    else:
        # Subsequent passes - use cached keys and values, only compute for new tokens
        prev_keys, prev_values = kv_cache
        new_keys, new_values = self.compute_kv(x[:, -1:])  # Only for the new token
        keys = torch.cat([prev_keys, new_keys], dim=1)
        values = torch.cat([prev_values, new_values], dim=1)
        return self.attention(x[:, -1:], keys, values), (keys, values)
```

**Benefits:**
- Eliminates redundant computations for previously processed tokens
- Reduces memory bandwidth requirements significantly
- Can improve token generation speed by 10-100x depending on sequence length

### Grouped-Query Attention (GQA)

<div align="center">
  <img src="https://pbs.twimg.com/media/FzjhZk5X0AYAs_-?format=jpg&name=4096x4096" alt="Grouped-Query Attention" width="700px">
  <p><em>Comparison of Multi-Head Attention, Multi-Query Attention, and Grouped-Query Attention</em></p>
</div>

Grouped-Query Attention (GQA) strikes an optimal balance between computational efficiency and model expressiveness:

**How It Works:**
1. Multiple query heads are grouped to share the same key and value projections
2. This reduces the KV cache size significantly compared to Multi-Head Attention
3. Maintains better quality than Multi-Query Attention by preserving some head diversity

```python
# Grouped-Query Attention implementation (simplified)
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, num_heads, kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.kv_heads = kv_heads  # kv_heads < num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim * kv_heads)  # Reduced projection size
        self.v_proj = nn.Linear(dim, self.head_dim * kv_heads)  # Reduced projection size
        
    def forward(self, x):
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.kv_heads, self.head_dim)
        
        # Duplicate k,v for heads that share the same k,v projection
        k = repeat_kv(k, self.num_heads // self.kv_heads)
        v = repeat_kv(v, self.num_heads // self.kv_heads)
        
        # Standard attention computation follows...
```

**Advantages:**
- Reduces memory requirements for the KV cache by a factor of (num_heads / kv_heads)
- Maintains most of the model quality of Multi-Head Attention
- Enables efficient inference on resource-constrained devices

### Rotary Positional Embeddings (RoPE)

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Z3uS6EvquUJIhCY-H-aeZA.png" alt="Rotary Positional Embeddings" width="600px">
  <p><em>Rotary positional embeddings apply rotation matrices to token embeddings</em></p>
</div>


Rotary Positional Embeddings (RoPE) provide a novel approach to encode position information directly into the attention mechanism:

**Mathematical Formulation:**

RoPE applies a rotation matrix $R_{\theta,m}$ to token embeddings, where $\theta$ determines the frequency and $m$ is the position:

$$R_{\theta,m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$$

The inner product between two rotated vectors naturally encodes their relative positions:

$$\langle R_{\theta,m}(q), R_{\theta,n}(k) \rangle = \langle q, R_{\theta,m-n}(k) \rangle$$

```python
# Simplified RoPE implementation
def apply_rotary_embeddings(q, k, positions):
    # Create sinusoidal patterns
    theta = 10000.0 ** (-torch.arange(0, dim, 2) / dim)
    positions = positions.reshape(-1, 1)  # [seq_len, 1]
    
    # Calculate sin/cos patterns
    freqs = positions * theta  # [seq_len, dim/2]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    # Reshape q and k for rotation
    q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
    
    # Apply rotation using broadcasting
    q_out = torch.stack([
        q_reshaped[..., 0] * cos - q_reshaped[..., 1] * sin,
        q_reshaped[..., 1] * cos + q_reshaped[..., 0] * sin
    ], dim=-1).flatten(-2)
    
    k_out = torch.stack([
        k_reshaped[..., 0] * cos - k_reshaped[..., 1] * sin,
        k_reshaped[..., 1] * cos + k_reshaped[..., 0] * sin
    ], dim=-1).flatten(-2)
    
    return q_out, k_out
```

**Key Benefits:**
- Preserves explicit relative position information between tokens
- Enables better extrapolation to sequence lengths not seen during training
- More computationally efficient than other positional embedding methods
- Handles long-range dependencies more effectively

### RMSNorm & SwiGLU Activation

<div align="center">
  <img src="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/swiglu.svg" alt="SwiGLU Activation Function" width="500px">
  <p><em>SwiGLU activation function in the feed-forward network</em></p>
</div>

#### RMSNorm (Root Mean Square Normalization)

RMSNorm simplifies traditional Layer Normalization while maintaining its stabilizing properties:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.scale
```

**Benefits:**
- Computationally simpler than LayerNorm (no mean subtraction)
- Stabilizes training and improves convergence
- Less sensitive to outliers in activations

#### SwiGLU Activation

SwiGLU (Swish-Gated Linear Unit) enhances the feed-forward network's expressiveness:

```python
def swiglu(x, w1, w2, w3):
    x1 = F.silu(x @ w1)  # Swish activation
    x2 = x @ w2
    return (x1 * x2) @ w3
```

**Advantages:**
- Smoother gradients than ReLU
- Better performance on language modeling tasks
- Enhanced representation capacity of the feed-forward network

---

## 📁 Code Structure

- **`model.py`:**  
  Implements the complete LLaMA 2 model architecture with detailed comments on each component. This file covers:
  - Multi-head attention with KV-cache support
  - Feed-forward network with SwiGLU activation
  - RMSNorm implementation
  - Rotary positional embeddings

- **`inference.py`:**  
  Demonstrates how to load a trained model and perform efficient inference:
  - Input preprocessing and tokenization
  - Autoregressive generation with KV-caching
  - Sampling strategies (greedy, top-k, top-p)
  - Output post-processing
---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/bikrammajhi/dl_from_scratch.git
cd dl_from_scratch/llama2

# Install requirements
pip install -r requirements.txt

# Run inference with the model
python inference.py --prompt "The meaning of life is" --max_tokens 100

```

### Example Usage

```python
import torch
from model import LLaMA2Model
from tokenizer import Tokenizer

# Load model and tokenizer
model = LLaMA2Model.from_pretrained("path/to/weights")
tokenizer = Tokenizer.from_pretrained("path/to/tokenizer")

# Tokenize input
prompt = "In a world where AI has become sentient,"
input_ids = tokenizer.encode(prompt)

# Generate text with KV-caching
generated_ids = model.generate(
    input_ids=torch.tensor([input_ids]),
    max_length=100,
    temperature=0.7,
    top_p=0.9
)

# Decode output
generated_text = tokenizer.decode(generated_ids[0].tolist())
print(generated_text)
```

---

## 🙏 Acknowledgements

This implementation is based on the LLaMA 2 paper by Meta AI. We'd like to thank the researchers and the open-source community for their valuable resources and discussions.

- [LLaMA 2 Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
