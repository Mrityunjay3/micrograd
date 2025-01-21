# Micrograd
Micrograd is a tiny autograd engine that I built following the extremely intuitive lecture by Andrej Karpathy. Building micrograd was the stepping stone for me in the world of Machine Learning and Artificial Intelligence. By doing this exercise I got a clear understanding of neural nets and backpropagation. After that I didn't look back and continued to upskill myself in all sorts of machine learning and data science techniques.


## Features

- **Core `Value` Class**: Handles data storage, gradients, and reverse-mode autodifferentiation.
- **Neuron, Layer, and MLP**: Implements the structure of neurons, layers, and a multi-layer perceptron.
- **Gradient Computation**: Supports operations like addition, subtraction, multiplication, division, exponentiation, and activation functions (e.g., `tanh` and `exp`).
- **Backpropagation**: Automatically computes gradients for loss using reverse-mode autodifferentiation.
- **Training Loop**: Demonstrates a basic example of training an MLP with mean squared error loss.

## Code Structure

1. **Value Class**: 
   - Represents individual units with `data`, `grad` (for gradients), and the ability to propagate gradients backward.
   - Supports mathematical operations such as addition, multiplication, and activation functions like `tanh` and `exp`.

2. **Neuron Class**:
   - Represents a single neuron with learnable weights and biases.
   - Uses the `tanh` activation function.

3. **Layer Class**:
   - Combines multiple neurons into a single layer.

4. **MLP Class**:
   - Implements a multi-layer perceptron with an arbitrary number of layers and neurons.

5. **Training Example**:
   - Demonstrates how to train the MLP using a sample dataset with mean squared error loss.

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib (optional, for future visualization)

Install dependencies using pip:

```bash
pip install numpy matplotlib
```


### Example

Below is a sample usage of the code:

```python
from micrograd import MLP, Value  

# Create an MLP with 3 inputs and 3 layers (4, 4, and 1 neuron per layer)
n = MLP(3, [4, 4, 1])

# Training data (inputs and outputs)
xs = [
    [-2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Training the MLP
for k in range(1000):
    # Forward pass
    ypred = [n(x) for x in xs]
    loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), Value(0.0))
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    
    # Gradient descent
    for p in n.parameters():
        p.data += -0.1 * p.grad
    
    if k % 100 == 0:
        print(f"Epoch {k}, Loss: {loss.data}")

# Predictions after training
print("Predictions:", [n(x).data for x in xs])
```
