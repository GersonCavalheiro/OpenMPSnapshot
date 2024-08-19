
#include "neural.hpp"


void nn::forward(void)
{
for (int layer = 1; layer < layers.size() - 1; layer += 1)
{
#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)                                           
{
double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
for (int synapse = 0; synapse < layers[layer - 1]; synapse += 1)                                    
{
REGISTER += weights[layer - 1][neuron][synapse] * a[layer - 1][synapse];                        
}
z[layer][neuron] = REGISTER;
}

#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layer] - 1; neuron += 1)
{
a[layer][neuron] = sigmoid(z[layer][neuron]);                                                       
}
}

#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
{
double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
{
REGISTER += weights[layers.size() - 2][neuron][synapse] * a[layers.size() - 2][synapse];            
}
z[layers.size() - 1][neuron] = REGISTER;
}

#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
{
a[layers.size() - 1][neuron] = sigmoid(z[layers.size() - 1][neuron]);                                   
}                                                                                                           
}
