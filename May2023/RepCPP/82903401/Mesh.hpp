#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef _WIN32
#include <time.h>
#else
#include <chrono>
#endif

extern size_t STEPS;
extern size_t MESH_SIZE;
extern size_t MESH_SIZE_EXTENDED;
extern int BLOCK_DIM_X;
extern int BLOCK_DIM_Y;

class SimpleTimer {
public:
#ifdef _WIN32
typedef std::chrono::high_resolution_clock clock;
#endif

SimpleTimer(std::string name) :
name(name) {
#ifndef _WIN32
clock_gettime(CLOCK_MONOTONIC, &begin);
#else
begin = clock::now();
#endif
}

~SimpleTimer(void) {
long long timeDiff;
#ifndef _WIN32
timespec end;
clock_gettime(CLOCK_MONOTONIC, &end);
timeDiff = (end.tv_sec - begin.tv_sec) * 1000 + (end.tv_nsec - begin.tv_nsec) / (1000 * 1000);
#else
timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - begin).count();
#endif

std::cout << "Execution of \"" << name << "\" took " << timeDiff << "ms" << std::endl;
}

private:
std::string name;
#ifndef _WIN32
timespec begin;
#else
clock::time_point begin;
#endif
};

class Mesh {
public:
typedef std::vector<std::vector<float> > ContainerType;

static float ENVIRONMENT_TEMP;
static float INITIAL_TEMP;
static ContainerType temperature;

void resize(size_t size) {
temperature = ContainerType(size, std::vector<float>(size, INITIAL_TEMP));
}

static float getTemperature(int x, int y) {
if (x < 0 || y < 0 || x >= temperature[0].size() || y >= temperature.size()) {
return ENVIRONMENT_TEMP;
}
return temperature[y][x];
}
};

template <typename T>
__host__ __device__ inline T* getElem(T* BaseAddress, size_t pitch, unsigned Row, unsigned Column) {
return reinterpret_cast<T*>(reinterpret_cast<char*>(BaseAddress) + Row * pitch) + Column;
}

float** allocMesh();
void freeMesh(float** mesh);
float* allocMeshLinear(size_t& pitch);
void validateResults(float** input);
void validateResults(float* input, size_t pitch);
void setValidateResults(bool value);
