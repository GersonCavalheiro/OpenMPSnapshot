#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>

using namespace  std::chrono;
using namespace cl::sycl;

__host__  void ReLU(float* input, uint size){
uint id = blockDim.x*blockIdx.x + threadIdx.x;
if(id<size){
float  element = input[id];
float sign = (int32_t(element)>>31) + 1.0f;
input[id] = sign * element;
}
}


__host__ void Leaky_ReLU(float* input, uint size){
uint id = blockDim.x*blockIdx.x + threadIdx.x;
if(id<size){
float  element = input[id];
float scale = 1.0f/2.0f;
float sign = (int32_t(element)>>31) + 2.0f;
input[id] = scale * sign * element;
}
}

void ReLU(float* input, int size){
default_selector deviceselector;
queue Queue(deviceselector);
buffer<float, 1> Buffer{input, sizeof(float )*size};
Queue.submit([&Buffer, size](handler& Handler){
auto accessor = Buffer.get_access<access::mode::write>(Handler);
Handler.parallel_for<class relu>(range<1>(size), [accessor](id<1> idx){
float element = accessor[idx[0]];
float sign = (int32_t(element)>>31) + 1.0f;
accessor[idx[0]] = sign * element;
});
});
}
void Leaky_ReLU(float* input, int size){
default_selector deviceselector;
queue Queue(deviceselector);
buffer<float, 1> Buffer{input, sizeof(float )*size};
Queue.submit([&Buffer, size](handler& Handler){
auto accessor = Buffer.get_access<access::mode::write>(Handler);
Handler.parallel_for<class leakyRelu>(range<1>(size), [accessor](id<1> idx){
float slope = 1.0f/2.0f;
float element = accessor[idx[0]];
float sign = (int32_t(element)>>31) + 2.0f;
accessor[idx[0]] = sign * element * slope;
});
});
}



void ReLU(float* input, int size){
#pragma omp simd
for(int i = 0; i<size; i++){
float element = *(input + i);
int sign = (int32_t(element)>>31) + 1.0f;
*(input + i) = sign * element;
}
}


void leaky_Relu(float* input, int size){
#pragma omp simd
for(int i = 0; i<size; i++){
float element = *(input+i);
float scaling_factor = 1.0f/2.0f;
auto sign =  float((int32_t(element)>>31))+2.0f;
*(input + i) = sign * element * scaling_factor;
}
}
