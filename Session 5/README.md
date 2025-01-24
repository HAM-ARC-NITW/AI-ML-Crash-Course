# Transformer Architecture

The **Transformer architecture** is a groundbreaking design for neural networks introduced in the paper _"Attention Is All You Need"_ by Vaswani et al. (2017). It has since become the backbone of modern natural language processing (NLP) and other sequence modeling tasks, thanks to its efficient and scalable mechanism for handling sequential data without relying on recurrent or convolutional layers.

---

## **Key Components of the Transformer Architecture**

A Transformer consists of an **encoder-decoder structure**, but many applications (e.g., BERT, GPT) focus on just one part (either the encoder or decoder). Here's how it works:

### **1. Encoder Block**

The encoder processes the input sequence and generates a contextual representation. It is composed of:

- **Input Embedding and Positional Encoding**: Since Transformers do not process input sequentially, positional information must be explicitly encoded. Positional encoding is added to word embeddings to provide a sense of order.
- **Self-Attention Mechanism**: The attention mechanism allows the model to focus on different parts of the input sequence dynamically.
- **Feed-Forward Network (FFN)**: A fully connected network applied to each token separately, with non-linear activations (e.g., ReLU).

Each encoder block follows this sequence:

- **Multi-Head Self-Attention**
- **Add & Layer Normalization (Residual Connection)**
- **Feed-Forward Network**
- **Add & Layer Normalization (Residual Connection)**

These steps are stacked multiple times (e.g., 6, 12, or more layers).

---

### **2. Decoder Block**

The decoder generates the output sequence, token by token, using the encoder’s output. It is similar to the encoder but includes additional components:

- **Masked Self-Attention Mechanism**: Prevents tokens from attending to future tokens by masking future positions in the sequence.

- **Encoder-Decoder Attention**: The decoder attends to the encoder’s output using a cross-attention mechanism to retrieve relevant context.

Each decoder block follows:

- **Masked Multi-Head Self-Attention**
- **Add & Layer Normalization (Residual Connection)**
- **Encoder-Decoder Attention**
- **Add & Layer Normalization (Residual Connection)**
- **Feed-Forward Network**
- **Add & Layer Normalization (Residual Connection)**

---

### **3. Final Linear Layer and Softmax**

For prediction tasks, the decoder output is passed through a linear layer and a softmax function to generate probabilities over the vocabulary.

---

## **Attention Mechanism in Transformers**

The **attention mechanism** is the core of the Transformer. It allows the model to selectively focus on relevant parts of the input sequence.

### **Scaled Dot-Product Attention**

Given a query vector \( Q \), key vector \( K \), and value vector \( V \):

1. Compute the dot product between \( Q \) and \( K^T \), scaled by \( \sqrt{d_k} \), where \( d_k \) is the dimension of the keys. This produces attention scores.

   ```math
   \text{Attention Scores} = \frac{Q K^T}{\sqrt{d_k}}
   ```

2. Apply the softmax function to normalize these scores into probabilities.

   ```math
   \text{Attention Weights} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)
   ```

3. Multiply the attention weights by \( V \) to produce the output.
   ```math
   \text{Output} = \text{Attention Weights} \cdot V
   ```

This process enables each token to attend to every other token in the sequence.

---

### **Multi-Head Attention**

Instead of applying attention once, Transformers use **multi-head attention** to capture information from different representation subspaces.

1. Input embeddings are linearly projected into multiple sets of \( Q, K, V \) matrices (typically 8 or 16 heads).
2. Each head computes scaled dot-product attention independently.
3. Outputs from all heads are concatenated and projected through another linear layer.

The formula:

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
```

Where

```math
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
```

**Benefits:**

- Multi-head attention allows the model to focus on different positions and capture richer representations.

---

### **Positional Encoding**

Transformers lack a sense of order in sequences, as they do not have recurrent structures. To address this, **positional encoding** is added to the embeddings. Commonly, sinusoidal functions are used:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)
```

Where \( pos \) is the position and \( i \) is the embedding index.

This encoding enables the model to learn positional relationships.

---

## **Advantages of Transformers**

1. **Parallelism**: Unlike RNNs, Transformers process entire sequences simultaneously, making them faster to train.
2. **Long-Range Dependencies**: The attention mechanism can model dependencies between distant tokens effectively.
3. **Scalability**: Transformers scale well with data and computational resources, enabling their application to large-scale models like GPT and BERT.

---

## **Applications of Transformers**

- **NLP**: Language models (GPT, BERT, T5), machine translation, summarization, question answering.
- **Vision**: Vision Transformers (ViT) for image classification.
- **Speech**: Speech recognition and synthesis.
- **Multimodal Tasks**: Combining text, images, and other modalities.

---

### **Diagram of a Transformer**

Below is a visual representation of the Transformer architecture:

![Transformer Architecture](.\TransformerArchitecturepng.png)
