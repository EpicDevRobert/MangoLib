#include <cmath>
#include <algorithm>
#include "aktywacje.h"

double mango::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}
double mango::sigmoid_derivative(double x) {
	return x * (1 - x);
}


double mango::relu(double x) {
	return fmax(0.0f, x);
}
double mango::relu_derivative(double x) {
	return (x > 0) ? 1 : 0;
}


double mango::leaky_relu(double x) {
	return (x > 0) ? x : 0.01 * x;
}
double mango::leaky_relu_derivative(double x) {
	return (x > 0) ? 1 : 0.01;
}