#include <cmath>
#include <iostream>
#include "../MangoStandard/standard.h"
#ifndef Deep
#include <vector>

namespace mango {
	double Cal_Layer(std::vector<neuron> &layer,int intex);
	class DeepNet {
	public:
		float learningRate = 0.1;
		std::vector<neuron> input;
		std::vector<std::vector<neuron>> hidden;
		std::vector<neuron> output;
		mango::aktywacje hidden_aktywacja = mango::aktywacje::sigmoidd;
		mango::aktywacje output_aktywacja = mango::aktywacje::sigmoidd;
		DeepNet(int inputsize,std::vector<int> sizeHidden,int sizeOutput) {
			input = std::vector<neuron>(inputsize, neuron(sizeHidden[0]));
			for (size_t i = 0; i < sizeHidden.size(); ++i) {
				if (i < sizeHidden.size() - 1) {
					hidden.push_back(std::vector<neuron>(sizeHidden[i], neuron(sizeHidden[i + 1])));
				}
				else {
					hidden.push_back(std::vector<neuron>(sizeHidden[i], neuron(sizeOutput)));
				}
			}
			output = std::vector<neuron>(sizeOutput, neuron(0));
		}
		void Run(std::vector<float> inputt);
		void Train(std::vector<float> inputt, std::vector<float> outputt);
	};
}

#define Deep
#endif // !Deep