

#include "../include/NeuralNet.hpp"





NeuralNet::NeuralNet(std::vector<unsigned int> topology, std::uniform_real_distribution<float> distr, std::mt19937 &generator)
: topology_{topology}, weights_{std::vector<Vector>()}, outputs_{std::vector<Vector>()}
{

bias_ = initializeRandom(topology.size() - 1, distr, generator);

for (unsigned int i = 0; i < topology.size(); i++)
{

if (i > 0)
{
weights_.push_back(initializeRandom(R * topology[i], distr, generator));
}

outputs_.push_back(Vector(topology[i]));
}
}



Vector NeuralNet::queryNet(const Vector &inputList, unsigned int p_mode)
{
outputs_[0] = inputList;

for (unsigned int i = 1; i < topology_.size(); i++)
{

float bias = bias_[i - 1]; 

if (p_mode == 0) 
{
for (unsigned int j = 0; j < topology_[i]; j++)
{
float acc = 0;
for (unsigned int offset = 0; offset < R; offset++)
{
acc += outputs_[i - 1][j + offset] * weights_[i - 1][j * R + offset];
}
outputs_[i][j] = activationFunction(acc + bias);
}
}
else if (p_mode == 1) 
{
#pragma omp parallel for num_threads(omp_get_max_threads())\
schedule(static) shared(outputs_, weights_, i, bias) default(none)
for (unsigned int j = 0; j < topology_[i]; j++)
{
float acc = 0;
for (unsigned int offset = 0; offset < R; offset++)
{
acc += outputs_[i - 1][j + offset] * weights_[i - 1][j * R + offset];
}
outputs_[i][j] = activationFunction(acc + bias);
}
}
else 
{
for (unsigned int j = 0; j < topology_[i]; j++)
{
float acc = 0;
#pragma omp parallel for reduction(+: acc) num_threads(omp_get_max_threads())\
schedule(static) shared(outputs_, weights_, i, j) default(none)
for (unsigned int offset = 0; offset < R; offset++)
{
acc += outputs_[i - 1][j + offset] * weights_[i - 1][j * R + offset];
}
outputs_[i][j] = activationFunction(acc + bias);
}
}
}

return outputs_.back();
}



Vector NeuralNet::initializeRandom(unsigned int n, std::uniform_real_distribution<float> distr, std::mt19937 &generator) const
{
Vector init(n);

for (unsigned int i = 0; i < n; i++)
{
init[i] = distr(generator);
}
return init;
}


float NeuralNet::activationFunction(float x) const
{
return 1. / (1. + std::exp(-x));
}
