#include<random>

class LinearRegression{
public:
LinearRegression(int input_size){
std::random_device device{};
std::normal_distribution<float> distribution{0, 2};

std::mt19937 generator{device()};

this->input_size = input_size;
this->bias = 1.0f;
this->weights = (float*)malloc(input_size * sizeof(float));
for (int i = 0; i < input_size; ++i) {
*(this->weights+i) = distribution(generator);
}
}

float forward(float* _input){
this->input = _input;
float result = 0.0f;
float intermediate = 0.0;
int i = 0;
auto model_weights = this->weights;
#pragma omp parallel default(none) private(intermediate) shared(result, model_weights, _input)
{
#pragma omp for
for (i = 0; i < this->input_size; ++i) {
intermediate = intermediate + *(this->weights + i) * *(_input + i);
}
#pragma omp atomic
result += intermediate;
}
result = result + this->bias;
return result;
}

void backward(float loss, float lr){
#pragma omp simd
for (int i = 0; i < this->input_size; ++i) {
*(this->weights + i)  = *(this->weights + i) - 2*lr*loss*(*(this->input + i));
}
this->bias = this->bias - 2*lr*loss;
}

private:
float* weights;
float  bias;
int input_size;
float* input;
};
