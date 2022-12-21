# Neural Network Mini-Library
Welcome to the Neural Network Mini-Library repository! This project is a low-level implementation of a multi-layered neural network, including a basic implementation of the backpropagation algorithm. It was created as part of the Introduction to Machine Learning module at Imperial College London. The library was tested using the Iris dataset.

## Features ## 
  - Implementation of a linear layer class
  - Implementation of activation function classes
    - Sigmoid
    - ReLU
    - Tanh
  - Implementation of a multi-layer network class
  - Implementation of a trainer class 
  - Implementation of a data preprocessing class

## Requirements ## 
  - NumPy

## Usage ## 
To use the mini-library, simply import the necessary classes and functions from the **`nn`** module.

```python
from nn import LinearLayer, ReLU, Sigmoid, Tanh, MultiLayerNetwork, Trainer, Preprocessor
```

### Linear Layer ###
To create a linear layer, simply instantiate the **`LinearLayer`** class with the appropriate input and output dimensions.

```python
linear_layer = LinearLayer(input_dim, output_dim)
```
You can then use the **`forward`** and **`backward`** methods to perform the forward and backward passes, respectively.

```python
output = linear_layer.forward(input)
linear_layer.backward(error)
```

### Activation Functions ###

The activation function classes ( **`ReLU`**,  **`Sigmoid`**, and  **`Tanh`**) can be used in the same way as the linear layer. Simply instantiate the class and use the **`forward`** and **`backward`** methods to perform the forward and backward passes.

```python
relu = ReLU()
output = relu.forward(input)
relu.backward(error)
```

### Multi-Layer Network ###
To create a multi-layer network, you can use the  **`MultiLayerNetwork`** class. Simply pass in a list of layers (linear or activation function layers) to the constructor.

```python
layers = [LinearLayer(input_dim, hidden_dim), ReLU(), LinearLayer(hidden_dim, output_dim)]
network = MultiLayerNetwork(layers)
```

You can then use the **`forward`** and **`backward`** methods to perform the forward and backward passes through the entire network.

```python
output = network.forward(input)
network.backward(error)
```

### Trainer ###
The **`Trainer class`** can be used to train the multi-layer network on a dataset. Simply instantiate the class with the network and the dataset, and then use the **`train`** method to perform the training.

```python
trainer = Trainer(network, dataset)
trainer.train(num_epochs)
```

### Preprocessor ###
The **`Preprocessor`** class can be used to preprocess the data before training. Simply instantiate the class with the dataset, and then use the **`preprocess`** method to perform the preprocessing.

```python
preprocessor = Preprocessor(dataset)
preprocessor.preprocess()
```


