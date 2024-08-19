

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include "neuron.h"


float evalNeuron(
size_t inputSize,
const float* input,
const float* weights,
float bias
)
{
#pragma omp simd reduction(+:bias) linear(input, weights) simdlen(8) aligned(weights)
for(size_t i = 0; i < inputSize; i++) {
bias += input[i] * weights[i];
}
return fmax(0.0, bias);
}
