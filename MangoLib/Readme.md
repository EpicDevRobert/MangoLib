MangoLib – A Lightweight Deep Neural Network Library
MangoLib is a compact and efficient C++ library designed for creating and training deep neural networks (DNNs). The library offers essential learning algorithms, including Adam and L2 regularization, and aims to provide a simple yet powerful foundation for building AI models. MangoLib is designed to be easy to use, extend, and integrate into your C++ projects.

Features
MangoLib currently supports the following learning methods and features:

Learning Methods:
Adam (Adaptive Moment Estimation)
Adam is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent. It computes adaptive learning rates for each parameter by considering both the first and second moments of the gradients. This makes Adam particularly well-suited for problems with large datasets and parameters.

L2 Regularization (Ridge Regression)
L2 regularization adds a penalty to the loss function, helping prevent overfitting by discouraging large weights. It is commonly used in deep learning to improve generalization.

Neural Network Structure:
Deep Neural Networks (DNN)
MangoLib allows you to create and train deep neural networks with multiple hidden layers. You can define the number of layers and neurons, enabling the creation of complex models suitable for a wide range of machine learning tasks.
Activation Functions:
Sigmoid
The sigmoid activation function maps any input value to a range between 0 and 1, making it suitable for binary classification tasks.

ReLU (Rectified Linear Unit)
ReLU sets all negative input values to 0, while leaving positive values unchanged. It is widely used in modern deep learning models due to its simplicity and efficiency in training.

Leaky ReLU
Leaky ReLU is a variant of ReLU that allows a small gradient for negative input values, which helps mitigate the problem of dying neurons in deep networks.

Benefits
Simplicity: MangoLib offers a minimalistic and intuitive interface for building and training neural networks. You can easily define the architecture, choose an optimizer, and train the model with just a few lines of code.
Efficiency: The library is optimized for performance and can handle large datasets with efficient memory usage.
Extensibility: While MangoLib currently offers the core features necessary for deep learning, it is designed in a modular way, making it easy to extend with new features, such as additional optimization algorithms or activation functions.
Lightweight: MangoLib is a small library with no external dependencies, making it ideal for integration into projects where minimalism is important.
Example Usage
#include <mango.h>
#include <iostream>
#include <cstdlib>
#include <vector>

int main() {
    // Create the neural network: 2 inputs, 5 hidden neurons, 1 output
    mango::net neuralNetwork(2, 5, 1);

    // Set the training method: SGD (Stochastic Gradient Descent)
    neuralNetwork.set_train(mango::trainMetods::SGD);

    // Training data: inputs and outputs
    std::vector<std::vector<float>> trainingData = {
        {0, 1, 1}, {0, 0, 0}, {1, 1, 0}, {1, 0, 1},
        {2, 0, 1}, {-1, 0, 1}, {0.1, 0, 1}, {-0.1, 0, 1}
    };

    // Training the network for 6000 epochs
    for (int epoch = 0; epoch < 6000; epoch++) {
        for (const auto& data : trainingData) {
            // Run the network with the given input
            neuralNetwork.Run({ data[0], data[1] });

            // Train the network using the input data and expected output
            neuralNetwork.Train({ data[0], data[1] }, { data[2] });
        }
    }

    // Testing section
    while (true) {
        float x, y;
        std::cout << "Enter two input values (x y): ";
        std::cin >> x >> y;

        // Run the network with new input values
        neuralNetwork.Run({ x, y });

        // Output the result from the network
        std::cout << "Output: " << neuralNetwork.return_output()[0] << std::endl;
    }
}

How to Install MangoLib
Clone the repository:


git clone https://github.com/yourusername/MangoLib.git
Build the project using CMake:

cd MangoLib
mkdir build
