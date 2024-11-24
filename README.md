**MangoLib - Lightweight Neural Networks**  

**Learning Methods:**  
- Backpropagation  
- SGD  

**Coming Soon:**  
- **Learning Methods:**  
  - Adam  
- **Activation Functions:**  
  - Tanh  
- **New Features:**  
  - Deep Neural Networks  

**Example Code:**  
#include <mango.h>
#include <iostream>
#include <vector>

int main() {
    mango::net neuralNetwork(2, 5, 1); // Create a network: 2 inputs, 5 hidden neurons, 1 output
    neuralNetwork.set_train(mango::trainMetods::SGD); // Set training method

    std::vector<std::vector<float>> trainingData = {
        {0, 1, 1}, {0, 0, 0}, {1, 1, 0}, {1, 0, 1}
    };

    // Train the network for 6000 epochs
    for (int epoch = 0; epoch < 6000; epoch++) {
        for (const auto& data : trainingData) {
            neuralNetwork.Run({ data[0], data[1] });
            neuralNetwork.Train({ data[0], data[1] }, { data[2] });
        }
    }

    // Test the network
    while (true) {
        float x, y;
        std::cout << "Enter two input values (x y): ";
        std::cin >> x >> y;

        neuralNetwork.Run({ x, y });
        std::cout << "Output: " << neuralNetwork.return_output()[0] << std::endl;
    }
}
```

---

**How to Install MangoLib:**  

1. Clone the repository:  
   git clone https://github.com/EpicDevRobert/MangoLib.git
   Or download it directly from Discord: MangoLib Discord https://discord.gg/tWrhVcMMsq

2. Build the project using CMake:  
   cd MangoLib
   mkdir build
   cd build
   cmake ..
   make
   ```  
