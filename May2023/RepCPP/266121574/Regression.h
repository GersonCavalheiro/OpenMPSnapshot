#ifndef SYCL_ML_LIB_REGRESSION_H
#define SYCL_ML_LIB_REGRESSION_H

#include<CL/sycl.hpp>
#include<SYCL/device_selector.h>

#include <random>
#include <vector>
#include <iostream>

#include "Kernels/RegressionKernel.h"
#include "Device.h"

using namespace cl::sycl;
using namespace std;

vector<size_t> getPowersofTwo(size_t input_size){
vector<size_t> powers;
while(input_size > 0){
powers.push_back(input_size % 2);
input_size = input_size/2;
}
return powers;
}

class LinearRegression{
public:
LinearRegression(size_t input_size){
std::random_device device{};
std::normal_distribution<float> distribution{0, 1};
std::ranlux48 generator{device()};

this->num_features = input_size;
this->bias = 0.5f;
this->weights = (float*)malloc(this->num_features * sizeof(float ));
#pragma omp parallel for
for(int i=0; i<input_size; i++){
*(this->weights + i) = distribution(generator);
}

this->use_sycl = 0;
}

void sycl(std::string Target_Device = " "){
this->use_sycl = 1;
this->selector = getDeviceSelector(Target_Device, this->selector);
this->Queue =  queue(*(*this->selector));
}

void setDevice(std::string Target_Device){
this->selector = getDeviceSelector(Target_Device, this->selector);
this->Queue = queue(*(*this->selector));
}

float forward(vector<float> input){
this->current_input = input.data();
float output = 0.0f;
if (this->use_sycl){
if(!(this->num_features & (this->num_features -1))) {
std::cout<<"Found true";
output = RegressionForward(input.data(), this->weights, this->bias, size_t(input.size()), this->Queue);
}

else{
vector<size_t> powers = getPowersofTwo(this->num_features);
float sum = 0.0f;
size_t offset = 0;
for(size_t power: powers){
std::vector<float> subvector_weight(power);
std::vector<float> subvector_input(power);
std::copy(this->weights+offset, this->weights + power, subvector_weight.begin());
std::copy(this->current_input + offset, this->current_input + power, subvector_input.begin());
sum += RegressionForward(subvector_input.data(), subvector_weight.data(), 0.0f , power, this->Queue);
offset = offset + power;
}
sum = sum + this->bias;
output = sum;
}
}

else{
float* _input = input.data();
output = 0.0f;
float intermediate = 0.0f;
int i =0;
#pragma omp parallel default(none) private(intermediate) shared(result, model_weights, _input)
{
#pragma omp for
for (i = 0; i < this->num_features; ++i) {
intermediate = intermediate + *(this->weights + i) * *(_input + i);
}
#pragma omp atomic
output += intermediate;
}
output = output + this->bias;
}

return output;
}
void backward(float loss, float learning_rate){
if(this->use_sycl){
updateWeights(this->weights, this->current_input, this->num_features, learning_rate, loss);
}
else{
#pragma omp parallel for

for (int i = 0; i < this->num_features; ++i) {
*(this->weights + i)  = *(this->weights + i) - 2*learning_rate*loss*(*(this->current_input + i));
}

}
this->bias = this->bias - 2*learning_rate*loss;
}


template<typename Func>
void backward(float loss, float learning_rate, Func loss_grad_func, float output, float target){
if(this->use_sycl){
updateWeights_custom(this->weights, this->current_input, this->num_features, learning_rate, loss, output, target, loss_grad_func);
}
else {
#pragma omp parallel for
for (int i = 0; i < this->num_features; ++i) {
*(this->weights + i)  = *(this->weights + i) - learning_rate * loss_grad_func(*(this->current_input + i), output, target);
}
}
this->bias = this->bias - learning_rate*loss_grad_func(this->bias, output, target);
}

void setWeights(float value){
vector<float> weight_vector(this->num_features, value);
std::copy(weight_vector.begin(), weight_vector.end(), this->weights);
}

float* getWeight() noexcept{
return this->weights;
}


private:
int num_features;
float* weights;
float bias;
float* current_input;
int use_sycl;
queue Queue;
unique_ptr<device_selector*> selector = std::make_unique<device_selector*>(new default_selector);
};

class LogisticRegression{
public:
LogisticRegression(size_t input_size){
std::random_device device{};
std::normal_distribution<float> distribution{0, 1};
std::ranlux48 generator{device()};

this->num_features = input_size;
this->bias = 0.5f;
this->weights = (float*)malloc(this->num_features * sizeof(float ));
#pragma omp parallel for
for(int i=0; i<input_size; i++){
*(this->weights + i) = distribution(generator);
}
this->use_sycl = 0;
}

void sycl(std::string Target_Device = " "){
this->selector = getDeviceSelector(Target_Device, this->selector);
this->Queue = queue(*(*this->selector));
this->use_sycl = 1;
}


template<typename Func>
float forward(vector<float> input, Func activation){
this->current_input = input.data();
float output = 0.0f;
if (this->use_sycl){
if(!(this->num_features & (this->num_features -1))) {
std::cout<<"Found true";
output = RegressionForward(input.data(), this->weights, this->bias, size_t(input.size()), this->Queue);
}

else{
vector<size_t> powers = getPowersofTwo(this->num_features);
float sum = 0.0f;
size_t offset = 0;
for(size_t power: powers){
std::vector<float> subvector_weight(power);
std::vector<float> subvector_input(power);
std::copy(this->weights+offset, this->weights + power, subvector_weight.begin());
std::copy(this->current_input + offset, this->current_input + power, subvector_input.begin());
sum += RegressionForward(subvector_input.data(), subvector_weight.data(), 0.0f , power, this->Queue);
offset = offset + power;
}
sum = sum + this->bias;
output = sum;
}
}

else{
float* _input = input.data();
output = 0.0f;
float intermediate = 0.0f;
int i =0;
#pragma omp parallel default(none) private(intermediate) shared(result, model_weights, _input)
{
#pragma omp for
for (i = 0; i < this->num_features; ++i) {
intermediate = intermediate + *(this->weights + i) * *(_input + i);
}
#pragma omp atomic
output += intermediate;
}
output = output + this->bias;
}

return activation(output);
}


template<typename Func>
void backward(float loss, float learning_rate, Func loss_grad_func, float output, float target){
if(this->use_sycl){
updateWeights_custom(this->weights, this->current_input, this->num_features, learning_rate, loss, output, target, loss_grad_func);
}
else {
#pragma omp parallel for
for (int i = 0; i < this->num_features; ++i) {
*(this->weights + i)  = *(this->weights + i) - learning_rate * loss_grad_func(*(this->current_input + i), output, target);
}
}
this->bias = this->bias - learning_rate*loss_grad_func(this->bias, output, target);
}

void setWeights(float value){
vector<float> weight_vector(this->num_features, value);
std::copy(weight_vector.begin(), weight_vector.end(), this->weights);
}

float* getWeight() noexcept{
return this->weights;
}


void setDevice(std::string Target_Device){
this->selector = getDeviceSelector(Target_Device, this->selector);
this->Queue = queue(*(*this->selector));
}

private:
int num_features;
float* weights;
float bias;
float* current_input;
int use_sycl;
queue Queue;
unique_ptr<device_selector*> selector = std::make_unique<device_selector*>(new default_selector);
};


#endif 

