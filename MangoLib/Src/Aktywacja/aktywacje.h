#ifndef aktywacje_main
#define aktywacje_main

namespace mango {
	enum aktywacje
	{
		sigmoidd = 0,
		ReLU = 1,
		leaky_ReLU = 2
	};
	double sigmoid(double x);
	double sigmoid_derivative(double x);
	double relu(double x);
	double relu_derivative(double x);
	double leaky_relu(double x);
	double leaky_relu_derivative(double x);

}
#endif // !aktywacje_main
