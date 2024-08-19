
#include "neural.hpp"


void nn::back_propagation(double* (&Y))
{
#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
{
delta[layers.size() - 2][neuron] = (a[layers.size() - 1][neuron] - Y[neuron]) * sig_derivative(a[layers.size() - 1][neuron]);               
}
#pragma omp parallel for num_threads(N_THREADS)
for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
{
double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)
{
REGISTER += weights[layers.size() - 2][neuron][synapse] * delta[layers.size() - 2][neuron];                                             
}
delta[layers.size() - 3][synapse] = REGISTER;
}

#pragma omp parallel for num_threads(N_THREADS)
for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)
{
delta[layers.size() - 3][synapse] = delta[layers.size() - 3][synapse] * sig_derivative(a[layers.size() - 2][synapse]);                      
}

for (int layer = 2; layer < layers.size() - 1; layer += 1)                                                                                      
{
#pragma omp parallel for num_threads(N_THREADS)
for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)
{
double REGISTER = 0.0;
#pragma omp simd reduction(+ : REGISTER)
for (int neuron = 0; neuron < layers[layers.size() - layer] - 1; neuron += 1)                                                           
{
REGISTER += weights[layers.size() - layer - 1][neuron][synapse] * delta[layers.size() - layer - 1][neuron];
}
delta[layers.size() - layer - 2][synapse] = REGISTER;
}

#pragma omp parallel for num_threads(N_THREADS)
for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)
{
delta[layers.size() - layer - 2][synapse] = delta[layers.size() - layer - 2][synapse] * sig_derivative(a[layers.size() - layer - 1][synapse]);
}                                                                                                                          
}
}


void nn::optimize(void)
{
#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layers.size() - 1]; neuron += 1)                                                                           
{
#pragma omp simd
for (int synapse = 0; synapse < layers[layers.size() - 2]; synapse += 1)                                                                    
{
weights[layers.size() - 2][neuron][synapse] -= LEARNING_RATE * delta[layers.size() - 2][neuron] * a[layers.size() - 2][synapse];        
}
}

for (int layer = 2; layer < layers.size(); layer += 1)                                                                                          
{
#pragma omp parallel for num_threads(N_THREADS)
for (int neuron = 0; neuron < layers[layers.size() - layer] - 1; neuron += 1)
{
#pragma omp simd
for (int synapse = 0; synapse < layers[layers.size() - layer - 1]; synapse += 1)                                                        
{
weights[layers.size() - layer - 1][neuron][synapse] -= LEARNING_RATE * delta[layers.size() - layer - 1][neuron] * a[layers.size() - layer - 1][synapse];
}
}
}
}
