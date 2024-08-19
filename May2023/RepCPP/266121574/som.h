#ifndef SYCL_ML_LIB_SOM_H
#define SYCL_ML_LIB_SOM_H
#endif 

#include <limits>


#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <atomic>
#include <cassert>
#include "Device.h"


class competitiveLearning{
public:
competitiveLearning(int feature_dim, int output_dim){
this->input_dim = feature_dim;
this->output_dim = output_dim;

std::random_device device{};
std::normal_distribution<float> distribution{0, 1};
std::ranlux48 generator{device()};

for(int i=0; i<output_dim; i++){
weights.push_back((float*)malloc(feature_dim * sizeof(float)));
#pragma omp parallel for
for(int j=0; j<this->input_dim; j++){
*(this->weights.at(i) + j) = distribution(generator);
}
}
}

template<typename criterion_func>
void forward(float* datapoint, float lr, criterion_func criterion){
int minPos = -1;
float minCriterion = std::numeric_limits<float>::max();


for(int i = 0; i<this->output_dim; i++){
auto distance = std::abs(criterion(datapoint, this->weights.at(i)));
if(distance < minCriterion){
minPos = i;
minCriterion = distance;
} 
}
assert(minPos != -1);

#pragma omp parallel for simd
for(int i=0; i<this->input_dim; i++)
*(this->weights.at(minPos) + i) = *(this->weights.at(minPos) + i) + lr*minCriterion;
}

private:
int input_dim;
int output_dim;
std::vector<float*> weights;
};