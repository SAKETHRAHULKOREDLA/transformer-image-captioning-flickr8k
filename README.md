# Transformer-based Image Captioning (Flickr8K)

This project implements an end-to-end image captioning system using Transformer architectures in PyTorch. The model generates natural language descriptions from images by combining visual features with sequence modeling.

## 🚀 Overview

The system uses a CNN-Transformer pipeline where image features are extracted using ResNet-50 and passed to a Transformer-based encoder–decoder to generate captions autoregressively.

## 🧠 Key Features

- Custom Transformer encoder–decoder using MultiheadAttention and LayerNorm
- Sinusoidal positional encoding
- ResNet-50 feature extraction for images
- Vocabulary construction and tokenization pipeline
- Autoregressive caption generation
- BLEU score evaluation
- Comparison with PyTorch Transformer and pretrained ViT-GPT2

## 🛠️ Pipeline

1. Load Flickr8K dataset
2. Build vocabulary and tokenize captions
3. Extract image features using ResNet-50
4. Train Transformer model
5. Generate captions
6. Evaluate using BLEU score

## 📊 Results

| Model | BLEU Score |
|------|-----------|
| Custom Transformer | 0.06 |
| PyTorch Transformer | 0.07 |
| ViT-GPT2 | 0.09 |

## 🔍 Observations

- The PyTorch Transformer slightly outperforms the custom implementation
- Pretrained ViT-GPT2 performs better due to large-scale pretraining
- BLEU scores are relatively low due to dataset size and use of greedy decoding

## ⚙️ Installation

```bash
pip install -r requirements.txt
