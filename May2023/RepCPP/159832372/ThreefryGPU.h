#include "Threefry.h"

#pragma once

__shared__ unsigned long long* Random123_Threefry_Device_d_Random;
__constant__ Threefry::Threefry123::key_type Random123_Threefry_Device_Key;

inline void Threefry::initDevice()
{
cudaMemcpyToSymbol(Random123_Threefry_Device_Key, &seed_, sizeof(seed_));
}

class Threefry::Device {
Threefry123::ctr_type m_ctr;
Threefry123 m_rng;

public:

__device__ Device(void* d, size_t indiv_id, Phase phase, int nb_indiv) {
m_ctr[0] = indiv_id + nb_indiv*phase;
Random123_Threefry_Device_d_Random = (unsigned long long*)d;
m_ctr.v[1] = ((unsigned long long*)d)[m_ctr[0]];
}

__device__ ~Device() {
Random123_Threefry_Device_d_Random[m_ctr[0]] = m_ctr[1];
}


__device__ unsigned long long random() {
++m_ctr[1];
return m_rng(m_ctr, Random123_Threefry_Device_Key)[0];
}

__device__ unsigned int random32() {
return random();
}


__device__ double randomDouble() {
return (random()&((1llu<<48)-1))/double(1llu<<48);
}

__device__ unsigned int random(unsigned int max) {
return randomDouble()*max;
}


__device__ float randomFloat() {
return (random32()&((1llu<<24)-1))/double(1llu<<24);
}

__device__ int32_t roulette_random(double* probs, int32_t nb_elts);
__device__ int32_t binomial_random(int32_t nb, double prob); 
};

class Threefry::DeviceCollectiveBlock {
Threefry123::ctr_type m_ctr;
Threefry123 m_rng;

public:

__device__ DeviceCollectiveBlock(void* d, size_t indiv_id, Phase phase, int nb_indiv) {
m_ctr[0] = indiv_id + nb_indiv*phase;
if(threadIdx.x == 0)
Random123_Threefry_Device_d_Random = (unsigned long long*)d;
m_ctr.v[1] = ((unsigned long long*)d)[m_ctr[0]]-blockDim.x+threadIdx.x+1;
}

__device__ ~DeviceCollectiveBlock() {
if(threadIdx.x == blockDim.x-1)
Random123_Threefry_Device_d_Random[m_ctr[0]] = m_ctr[1];
}


__device__ unsigned long long random() {
m_ctr[1] += blockDim.x;
return m_rng(m_ctr, Random123_Threefry_Device_Key)[0];
}

__device__ unsigned int random32() {
return random();
}


__device__ double randomDouble() {
return (random()&((1llu<<48)-1))/double(1llu<<48);
}

__device__ unsigned int random(unsigned int max) {
return randomDouble()*max;
}


__device__ float randomFloat() {
return (random32()&((1llu<<24)-1))/double(1llu<<24);
}

__device__ int32_t roulette_random(double* probs, int32_t nb_elts);
__device__ int32_t binomial_random(int32_t nb, double prob); 
};
