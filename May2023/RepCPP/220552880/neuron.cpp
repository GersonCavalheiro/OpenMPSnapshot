


#include <cmath>

#include "neuron.h"


float evalNeuron(
size_t inputSize,
const float *input,
const float *weights,
float bias
)
{
float x = bias;

#pragma omp simd reduction(+:x) aligned(input, weights)
for (size_t i = 0; i < inputSize; i++)
{
x += input[i] * weights[i];
}

return fmaxf(.0f, x);
}
