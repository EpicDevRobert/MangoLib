#ifndef MangoStandard
#define MangoStandard
#include "../Aktywacja/aktywacje.h"
#include "../Train/TrainMetods.h"
#include <vector>
#include <cstdlib> 

namespace mango {
	class neuron {
	public:
		float bias = 0;
		std::vector<float> weight;
		float sum = 0, delta = 0;
		neuron(int weigths) {
			for (int i = 0; i < weigths; i++) {
				float rande = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
				weight.push_back(rande);
			}
		}
	};
	class net {
	public:
		std::vector<neuron> input;
		std::vector<neuron> hidden;
		std::vector<neuron> output;
		mango::aktywacje hidden_aktywacja = mango::aktywacje::sigmoidd;
		mango::aktywacje output_aktywacja = mango::aktywacje::sigmoidd;
		mango::trainMetods trainerr = mango::trainMetods::Backpropagation;
		float LearningRate = 0.1;
		float lambdaa = 0.1f;
		net(int inpute, int hiddene, int outpute) {
			input = std::vector<neuron>(inpute, neuron(hiddene));
			hidden = std::vector<neuron>(hiddene, neuron(outpute));
			output = std::vector<neuron>(outpute, neuron(0));
		}
		net(int inpute, int hiddene, int outpute, aktywacje output_aktywacja, aktywacje hidden_aktywacja) :output_aktywacja(output_aktywacja), hidden_aktywacja(hidden_aktywacja) {
			input = std::vector<neuron>(inpute, neuron(hiddene));
			hidden = std::vector<neuron>(hiddene, neuron(outpute));
			output = std::vector<neuron>(outpute, neuron(0));
		}
		void Run(std::vector<float> inputs);
		void Train(std::vector<float> inputs, std::vector<float> target);
		std::vector<float> return_output();
		void set_train(mango::trainMetods trainer);
	};
}
#endif // !MangoStandard
