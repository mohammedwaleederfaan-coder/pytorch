# 🔥 PyTorch Fundamentals

A hands-on notebook covering the core concepts of PyTorch — from tensors to building and training a neural network with GPU support.

---

## 📚 Topics Covered

### 1. Tensors
- Creating tensors from lists, NumPy arrays, and built-in functions (`zeros`, `ones`, `rand`)
- Tensor attributes: `shape`, `dtype`, `device`, `numel()`
- Operations: addition, subtraction, multiplication, division, matrix multiplication
- Reshaping, slicing, and indexing

### 2. Autograd
- `requires_grad=True` and how PyTorch tracks gradients
- Computing gradients with `.backward()`
- Understanding `x.grad` and the chain rule in practice

### 3. Building a Neural Network
- Subclassing `nn.Module`
- Defining layers with `nn.Linear` and activation functions (`ReLU`, `Sigmoid`)
- Forward pass implementation
- Loss function: `BCELoss`
- Optimizers: `Adam`
- Full training loop from scratch

### 4. Best Practices
- Train/Validation split with `random_split`
- Batch training with `DataLoader` and `TensorDataset`
- Learning rate scheduling with `StepLR`
- GPU support with `torch.cuda`
- Detecting overfitting via Train vs Val Loss

---

## 🛠️ Requirements

```bash
pip install torch numpy scikit-learn
```

---

## 🚀 How to Run

```bash
git clone https://github.com/mohammedwaleederfaan-coder/pytorch-fundamentals.git
cd pytorch-fundamentals
jupyter notebook pytorch-fundamentals.ipynb
```

---

## 📁 Structure

```
pytorch-fundamentals/
│
├── pytorch-fundamentals.ipynb   # Main notebook
└── README.md
```

---

## 🧠 What's Next

This notebook is part of a series:

| Notebook | Status |
|----------|--------|
| PyTorch Fundamentals | ✅ Done |
| PyTorch NLP | 🔜 Coming Soon |

---

## 👤 Author

**Mohammed Waleed**
- GitHub: [mohammedwaleederfaan-coder](https://github.com/mohammedwaleederfaan-coder)
- LinkedIn: [mohammed-waleed-0065b9409](https://linkedin.com/in/mohammed-waleed-0065b9409)
