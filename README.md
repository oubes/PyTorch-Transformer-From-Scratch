# PyTorch Transformer From Scratch 🚀

An end-to-end implementation of the original Transformer architecture as described in the seminal paper **"Attention Is All You Need" (2017)**. This project focuses on building the entire mechanism from the ground up using **PyTorch 2.7.0**.


<p align="center">
  <img src="https://github.com/user-attachments/assets/2e5c81e0-bd95-4de3-b5b7-d7da2a8345c6" width="340">
</p>

---

## 📌 Project Overview

This repository provides a clean, modular, and well-commented implementation of the Transformer model. It is designed for those who want to understand the inner workings of Self-Attention, Multi-Head Attention, and the Encoder-Decoder dynamics without relying on high-level black-box libraries.

### Key Features:
* **Full Architecture**: Implementation of `InputEmbeddings`, `PositionalEncoding`, `MultiHeadAttention`, and `FeedForward` blocks.
* **Modern Stack**: Optimized for **PyTorch 2.7.0** and **Transformers 4.52.4**.
* **Data Integration**: Ready-to-use pipeline with HuggingFace `datasets` and `tokenizers`.
* **Hardware Accelerated**: Support for Training/Inference on CUDA devices.

---

## 🏗️ Architecture Visualization

The model architecture implemented here follows the standard structure:



---

## 🚀 Getting Started

### 1. Installation

This project requires specific versions of dependencies to ensure compatibility with 3D and Vision processing libraries. 

```bash
# Clone the repository
git clone https://github.com/oubes/PyTorch-Transformer-From-Scratch.git
cd PyTorch-Transformer-From-Scratch

# Install the required dependencies
pip install -r requirements.txt

```

### 2. Core Dependencies

The project relies on these primary libraries (based on your environment):

* **Core**: `torch==2.7.0`, `numpy==1.26.2`
* **NLP**: `transformers==4.52.4`, `tokenizers==0.21.1`, `datasets`
* **Utilities**: `tqdm`, `PyYAML`, `scipy`

---

## 📁 Repository Structure

```text
├── src/
│   ├── model.py        # Core Transformer, Encoder, Decoder classes
│   ├── dataset.py      # Custom Dataset and Tokenizer logic
│   ├── config.py       # Model hyperparameters and training config
│   └── train.py        # Training loop and Validation logic
├── notebooks/          # Step-by-step interactive walkthroughs
├── requirements.txt    # Full environment specifications
└── README.md           # Project documentation

```

---

## 🛠️ Usage

To start training the model on the default dataset (OpusBooks English-Italian):

```python
python src/train.py

```

---

## 📈 Learning Path

While building this, I explored:

1. **Scaled Dot-Product Attention**: Why we scale by .
2. **Masking**: Implementing causal masks for the Decoder to prevent "cheating" during training.
3. **Residual Connections**: How they help in training deep networks (6+ layers).

---

## ✍️ Credits

* **Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* **Reference**: Inspired by the architectural breakdown by Umar Jamil.

---

## 📄 License

MIT License. Feel free to use and modify!

