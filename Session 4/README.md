# Fully Connected (Linear) Layer

A **Fully Connected Layer** (also known as a **Linear Layer**) is a fundamental component of artificial neural networks. It is called "fully connected" because each neuron in the layer is connected to every neuron in the previous layer. This architecture enables the network to learn complex relationships within the data by allowing information to flow freely between layers.

## Mathematical Representation

In a fully connected layer, the output can be mathematically expressed as:

$$
\mathbf{y} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}
$$

### Components

- **Output Vector** ($\mathbf{y}$): This is the result of the transformation applied to the input vector by the layer.
- **Weight Matrix** ($\mathbf{W}$): This matrix contains the weights that determine the strength of the connections between the input features and the neurons in the layer. Each entry $w_{ij}$ represents the weight connecting the $j^{th}$ input to the $i^{th}$ neuron.
- **Input Vector** ($\mathbf{x}$): This vector consists of the input features fed into the layer.
- **Bias Vector** ($\mathbf{b}$): This vector adds an additional degree of freedom to the model. Each neuron has its bias term, which allows the model to fit the data better.

### General Dimension Expressions

1. **For Single Input:**

   - If the input vector $\mathbf{x}$ has a shape of $(n, 1)$:
     - Input: $\mathbf{x} \in \mathbb{R}^{n \times 1}$
     - Weight matrix: $\mathbf{W} \in \mathbb{R}^{m \times n}$
     - Bias vector: $\mathbf{b} \in \mathbb{R}^{m \times 1}$
     - Output: $\mathbf{y} \in \mathbb{R}^{m \times 1}$

2. **For Batched Input:**
   - If the input matrix $\mathbf{X}$ has a shape of $(b, n)$, where $b$ is the batch size:
     - Input: $\mathbf{X} \in \mathbb{R}^{b \times n}$
     - Weight matrix: $\mathbf{W} \in \mathbb{R}^{m \times n}$
     - Bias vector: $\mathbf{b} \in \mathbb{R}^{m \times 1}$
     - Output: $\mathbf{Y} \in \mathbb{R}^{b \times m}$

### Example

Consider a specific example where:

- An input vector $\mathbf{x}$ has the shape $(3, 1)$:

$$
\mathbf{x} = \begin{pmatrix}
x_1 \\
x_2 \\
x_3
\end{pmatrix}
$$

- A weight matrix $\mathbf{W}$ has the shape $(2, 3)$:

$$
\mathbf{W} = \begin{pmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23}
\end{pmatrix}
$$

- A bias vector $\mathbf{b}$ has the shape $(2, 1)$:

$$
\mathbf{b} = \begin{pmatrix}
b_1 \\
b_2
\end{pmatrix}
$$

The output $\mathbf{y}$ will be computed as follows:

$$
\mathbf{y} = \mathbf{W} \cdot \mathbf{x} + \mathbf{b}
$$

Performing the matrix multiplication and addition gives:

$$
\mathbf{y} = \begin{pmatrix}
w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \\
w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2
\end{pmatrix}
$$

In this example, $\mathbf{y}$ will have the shape $(2, 1)$, indicating that there are two neurons in this layer.

## Activation Functions

Activation functions are essential for introducing non-linearity into neural networks, enabling them to learn and approximate complex functions. Without activation functions, the entire neural network would behave like a linear model, severely limiting its capacity to solve complex problems.

### 1. Sigmoid Activation Function

The **Sigmoid** function maps any real-valued number into the range (0, 1). It is particularly useful for binary classification problems.

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Properties

- **Range**: (0, 1)
- **Derivative**: The derivative of the sigmoid function is given by:

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### 2. ReLU (Rectified Linear Unit)

The **ReLU** function is one of the most widely used activation functions in deep learning. It outputs the input directly if it is positive; otherwise, it outputs zero.

$$
\text{ReLU}(x) = \max(0, x)
$$

#### Properties

- **Range**: [0, ∞)
- **Derivative**: The derivative of ReLU is:

$$
\text{ReLU}'(x) =
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

### 3. Tanh (Hyperbolic Tangent)

The **Tanh** function is similar to the sigmoid function but maps input to the range (-1, 1), making it zero-centered.

$$
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

#### Properties

- **Range**: (-1, 1)
- **Derivative**: The derivative of the tanh function is given by:

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

## Loss Functions

Loss functions are critical for training neural networks, as they quantify the difference between the predicted output and the actual output. During training, the model optimizes its parameters to minimize the loss function, improving its predictive accuracy.

### 1. Mean Squared Error (MSE)

For regression tasks, the **Mean Squared Error** is defined as:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:

- $y_i$ is the true value.
- $\hat{y}_i$ is the predicted value.
- $n$ is the number of observations.

### 2. Binary Cross-Entropy (BCE)

For binary classification problems, the **Binary Cross-Entropy** loss function measures the performance of a classification model whose output is a probability value between 0 and 1.

$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

Where:

- $y_i$ is the true label (0 or 1).
- $\hat{y}_i$ is the predicted probability.

### 3. Categorical Cross-Entropy (CCE)

For multi-class classification problems, the **Categorical Cross-Entropy** loss function is defined as:

$$
\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

Where:

- $C$ is the number of classes.
- $y_i$ is the true probability distribution (one-hot encoded).
- $\hat{y}_i$ is the predicted probability distribution.
