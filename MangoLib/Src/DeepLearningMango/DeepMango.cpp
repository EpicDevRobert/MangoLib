#include <vector>
#include "aktywacje.h"
#include "DeepMango.h"
#include <cmath>
double mango::Cal_Layer(std::vector<neuron>& layer, int intex) {
	float sum = 0;
	for (int i = 0; i < layer.size();i++) {
		sum += layer[i].sum * layer[i].weight[intex];
	}
	return sum;
}
void mango::DeepNet::Run(std::vector<float> inputt) {
	for (int i = 0; i < inputt.size();i++) {
		input[i].sum = inputt[i];
	}
	for (int i = 0; i < hidden[0].size(); i++) {
		float sum = mango::Cal_Layer(input, i) + hidden[0][i].bias;
		hidden[0][i].sum = mango::sigmoid(sum);
	}
	for (size_t i = 1; i < hidden.size(); i++) {
		for (size_t j = 0; j < hidden[i].size(); j++) {
			float sum = Cal_Layer(hidden[i - 1], j) + hidden[i][j].bias;
			hidden[i][j].sum = mango::sigmoid(sum);
		}
	}
	for (size_t i = 0; i < output.size(); i++) {
		float sum = Cal_Layer(hidden.back(), i) + output[i].bias;
		output[i].sum = mango::sigmoid(sum);
	}
}

void mango::DeepNet::Train(std::vector<float> inputt, std::vector<float> outputt) {
    // 1. Ustawienie wartoœci wejœæ
    for (size_t i = 0; i < inputt.size(); ++i) {
        input[i].sum = inputt[i];
    }

    // 2. Obliczanie b³êdów dla warstwy wyjœciowej (output)
    for (int i = 0; i < output.size(); i++) {
        float error = output[i].sum - outputt[i]; // Obliczanie b³êdu
        output[i].delta = error * mango::sigmoid_derivative(output[i].sum); // Obliczanie delty dla warstwy wyjœciowej
    }

    for (int i = hidden.size() - 1; i >= 0; i--) {
        for (int j = 0; j < hidden[i].size(); j++) {
            float error = 0;
            if (i == hidden.size() - 1) { 
                for (int k = 0; k < output.size(); k++) {
                    error += output[k].delta * hidden[i][j].weight[k]; 
                }
            }
            else {
                for (int k = 0; k < hidden[i + 1].size(); k++) {
                    error += hidden[i + 1][k].delta * hidden[i][j].weight[k];
                }
            }
            hidden[i][j].delta = error * mango::sigmoid_derivative(hidden[i][j].sum);
        }
    }

    for (int i = 0; i < input.size(); i++) {
        float error = 0;
        for (int j = 0; j < hidden[0].size(); j++) {
            error += hidden[0][j].delta * hidden[0][j].weight[i];
        }
        input[i].delta = error * mango::sigmoid_derivative(input[i].sum);
    }


    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < hidden[hidden.size() - 1].size(); j++) {
            output[i].weight[j] += learningRate * output[i].delta * hidden[hidden.size() - 1][j].sum;
        }
        output[i].bias += learningRate * output[i].delta;
    }

    
    for (int i = hidden.size() - 1; i >= 0; i--) { 
        for (int j = 0; j < hidden[i].size(); j++) {
            if (i == 0) {
                for (int k = 0; k < input.size(); k++) {
                    hidden[i][j].weight[k] += learningRate * hidden[i][j].delta * input[k].sum; 
                }
            }
            else {
                for (int k = 0; k < hidden[i - 1].size(); k++) {
                    hidden[i][j].weight[k] += learningRate * hidden[i][j].delta * hidden[i - 1][k].sum;
                }
            }
            hidden[i][j].bias += learningRate * hidden[i][j].delta;
        }
    }

    for (int i = 0; i < hidden[0].size(); i++) {
        for (int j = 0; j < input.size(); j++) {
            hidden[0][i].weight[j] += learningRate * hidden[0][i].delta * input[j].sum;        }
    }

}

