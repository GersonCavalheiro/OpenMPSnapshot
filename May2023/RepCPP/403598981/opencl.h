#pragma once
#include "standard.h"
#include <chrono>
using namespace std::chrono;
#include "CL/cl.hpp"


namespace dijkstra {
class opencl: public standard
{
public:
opencl(int V);
~opencl();
void printSolution(int dist[], int n);
void flatArray(int** array, int* flat_array, int width, int height);
void dijkstra(int **graph, int src);
cl::Kernel createKernel(cl::Device default_device, cl::Context context, std::string kernel, char* kernelName);
cl::Device getOCLDevice();
};
}
