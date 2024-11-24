#include "standard.h"
#include <cmath>
#include <iostream>
void mango::net::Run(std::vector<float> inputs) {
	if (inputs.size() == input.size()) {
		for (int i = 0; i < inputs.size(); i++) {
			input[i].sum = inputs[i];
		}

		for (int j = 0; j < hidden.size(); j++) {
			float sum = 0;
			for (int i = 0; i < inputs.size(); i++) {
				sum += input[i].weight[j] * input[i].sum;
			}
			sum += hidden[j].bias;
			if (hidden_aktywacja == sigmoidd) {
				hidden[j].sum = mango::sigmoid(sum);
			}
			else if (hidden_aktywacja == ReLU) {
				hidden[j].sum = mango::relu(sum);
			}
			else if (hidden_aktywacja == leaky_ReLU) {
				hidden[j].sum = mango::leaky_relu(sum);
			}
		}

		for (int i = 0; i < output.size(); i++) {
			float sum = 0;
			for (int j = 0; j < hidden.size(); j++) {
				sum += hidden[j].weight[i] * hidden[j].sum;
			}
			sum += output[i].bias;
			if (output_aktywacja == sigmoidd) {
				output[i].sum = mango::sigmoid(sum);
			}
			else if (output_aktywacja == ReLU) {
				output[i].sum = mango::relu(sum);
			}
			else if (output_aktywacja == leaky_ReLU) {
				output[i].sum = mango::leaky_relu(sum);
			}
		}
	}
	else {
		std::cerr << "dodaj ta sama liczbe inputow" << std::endl;
		std::cin.get();
	}
}

void mango::net::Train(std::vector<float> inputs, std::vector<float> target) {
	if (inputs.size() == input.size() && target.size() == output.size()) {
		for (int i = 0; i < output.size(); i++) {
			float error = target[i] - output[i].sum;
			if (output_aktywacja == sigmoidd) {
				output[i].delta = error * mango::sigmoid_derivative(output[i].sum);
			}
			else if (output_aktywacja == ReLU) {
				output[i].delta = error * mango::relu_derivative(output[i].sum);
			}
			else if (output_aktywacja == leaky_ReLU) {
				output[i].delta = error * mango::leaky_relu_derivative(output[i].sum);
			}
		}
		for (int i = 0; i < hidden.size(); i++) {
			float error = 0;
			for (int j = 0; j < output.size(); j++) {
				error += output[j].delta * hidden[i].weight[j];
			}
			if (hidden_aktywacja == sigmoidd) {
				hidden[i].delta = error * mango::sigmoid_derivative(hidden[i].sum);
			}
			else if (hidden_aktywacja == ReLU) {
				hidden[i].delta = error * mango::relu_derivative(hidden[i].sum);
			}
			else if (hidden_aktywacja == leaky_ReLU) {
				hidden[i].delta = error * mango::leaky_relu_derivative(hidden[i].sum);
			}
		}
		if (trainerr == mango::Backpropagation) {
			for (int i = 0; i < output.size(); i++) {
				for (int j = 0; j < hidden.size(); j++) {
					hidden[j].weight[i] += LearningRate * output[i].delta * hidden[j].sum;
				}
				output[i].bias += LearningRate * output[i].delta;
			}

			for (int j = 0; j < hidden.size(); j++) {
				for (int i = 0; i < input.size(); i++) {
					input[i].weight[j] += LearningRate * hidden[j].delta * input[i].sum;
				}
				hidden[j].bias += LearningRate * hidden[j].delta;
			}
		}
		else if (trainerr == mango::trainMetods::SGD) {
			for (int i = 0; i < output.size(); i++) {
				for (int j = 0; j < hidden.size(); j++) {
					hidden[j].weight[i] += LearningRate * output[i].delta * hidden[j].sum;
				}
				output[i].bias += LearningRate * output[i].delta;
			}

			for (int i = 0; i < hidden.size(); i++) {
				for (int j = 0; j < input.size(); j++) {
					input[j].weight[i] += LearningRate * hidden[i].delta * input[j].sum;
				}
				hidden[i].bias += LearningRate * hidden[i].delta;
			}
		}

		else {
			std::cerr << "blad wynika z target lub inputs" << std::endl;
			std::cin.get();
		}
	}
}
std::vector<float> mango::net::return_output() {
	std::vector<float> dat;
	for (int i = 0; i < output.size(); i++) {
		dat.push_back(output[i].sum);
	}
	return dat;
}
void mango::net::set_train(mango::trainMetods trainer) {
	trainerr = trainer;
}