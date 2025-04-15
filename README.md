# LLaMA 2 from Scratch

This repository provides an end-to-end implementation of the **LLaMA 2** model—a variant of the Generative Pretrained Transformer (GPT) architecture developed by Meta AI. The code emphasizes a clear exposition of the model architecture and the inference process, featuring extensive comments and a modular structure to aid understanding and experimentation.

## Table of Contents

- [Model Features](#model-features)
- [Implementation Highlights](#implementation-highlights)
  - [KV-Caching for Efficient Inference](#kv-caching-for-efficient-inference)
  - [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
  - [Rotary Positional Embeddings (RoPE)](#rotary-positional-embeddings-rope)
- [Code Structure](#code-structure)

---

## Model Features

- **RMS-Normalization:**  
  RMSNorm simplifies the traditional Layer Normalization by stabilizing layer activations, which mitigates the internal covariate shift and improves model convergence. This technique has proven particularly effective in the LLaMA 2 architecture.

- **Activation Function – SwiGLU:**  
  Instead of using ReLU, LLaMA 2 leverages the SwiGLU activation function. This choice enhances the model's training performance while providing a smoother gradient flow.

- **Rotary Positional Embeddings (RoPE):**  
  Inspired by innovations from the GPT-Neo-X project, RoPE integrates rotational matrices into the embedding process, enriching the model's positional awareness and contextual understanding.

- **Increased Context Length & Grouped-Query Attention (GQA):**  
  The model doubles the context window from 2048 to 4096 tokens. GQA further refines the attention mechanism by reducing computational redundancy, enabling efficient processing of long documents, chat histories, and summarization tasks.

---

## Implementation Highlights

### KV-Caching for Efficient Inference

KV-Caching is a key optimization that accelerates inference during autoregressive decoding. In this process:

- **Autoregressive Decoding:**  
  Each token is predicted using only prior tokens, ensuring the self-attention mechanism remains causal.

- **Storing Key/Value Projections:**  
  By caching the results of the key and value projections, the model avoids redundant computations, which drastically speeds up subsequent token generation.

- **Efficiency Boost:**  
  This caching mechanism not only improves throughput but also optimizes memory usage during inference.

![KV-Caching Diagram](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/kv-cache-optimization.png)

---

### Grouped-Query Attention (GQA)

Grouped-Query Attention (GQA) is a refined version of Multi-Query Attention (MQA) originally proposed by Shazeer (2019):

- **Computational Efficiency:**  
  Unlike traditional multi-head attention—which replicates the entire computation for each attention head—GQA reduces redundancy by sharing the key and value transformations among heads.

- **Memory Optimization:**  
  By cutting down on the volume of data processed and stored, GQA decreases memory usage, especially for the KV-cache, while keeping the arithmetic intensity high.

- **Enhanced Performance:**  
  This improvement sustains similar accuracy levels to Multi-Head Attention (MHA) but with enhanced efficiency, making it ideal for large-scale models like LLaMA 2.

![Grouped-Query Attention](https://pbs.twimg.com/media/FzjhZk5X0AYAs_-?format=jpg&name=4096x4096)

---

### Rotary Positional Embeddings (RoPE)

RoPE is a novel technique to embed positional information within token representations:

- **Contextual Relevance:**  
  Traditional attention mechanisms benefit from knowing a token’s position. While absolute embeddings denote the exact position, relative embeddings capture the distance between tokens. RoPE marries both ideas.

- **Rotational Matrices:**  
  By integrating the position-dependent rotation matrices into the embedding process, RoPE ensures that the inner product between query and key vectors reflects their relative positions—enhancing the overall attention mechanism.

- **Improved Token Relationships:**  
  This approach refines how the model perceives context, resulting in better handling of long-range dependencies and improved attention across tokens.

![Rotary Positional Embeddings](https://pbs.twimg.com/media/FrqjrsmXoAQhr2R.jpg)

---

## Code Structure

- **`model.py`:**  
  Implements the LLaMA transformer model with detailed comments on each component. This file covers the construction of the model's layers, normalization techniques, and attention mechanisms.

- **`inference.py`:**  
  Demonstrates how to load a trained model and perform inference. This script provides insights into input preprocessing, token generation, and post-processing of outputs.

---
