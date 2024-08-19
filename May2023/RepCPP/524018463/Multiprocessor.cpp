#include "Multiprocessor.h"
#include <omp.h>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>

HpcSamples::Multiprocessor::Multiprocessor(uint16_t size)
{
std::cout << "Matrix size: " << size << "x" << size << std::endl;

_size = size;

_matrixA = new float*[size];
_matrixB = new float*[size];
_matrixC = new float*[size];

for (int i = 0; i < size; i++)
{
_matrixA[i] = new float[size];
_matrixB[i] = new float[size];
_matrixC[i] = new float[size];
}

Initialize(_size);
}

HpcSamples::Multiprocessor::~Multiprocessor()
{
for (int i = 0; i < _size; i++)
{
delete _matrixA[i];
delete _matrixB[i];
delete _matrixC[i];
}

delete[] _matrixA;
delete[] _matrixB;
delete[] _matrixC;
}

void HpcSamples::Multiprocessor::Initialize(uint16_t size)
{
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; j++)
{
_matrixA[i][j] = 1.0f;
_matrixB[i][j] = 0.1f;
}
}
}

bool HpcSamples::Multiprocessor::CheckResult()
{
float expected = 0.1f * _size;

for (int i = 0; i < _size; i++)
{
for (int j = 0; j < _size; j++)
{
float diff = fabsf(expected - _matrixC[i][j]);
if (diff > FLT_EPSILON) return false;
}
}

return true;
}

void HpcSamples::Multiprocessor::RunOnMultiprocessor()
{
std::cout << std::endl;
std::cout << "----------" << std::endl;
std::cout << "Matrix Multiplication on CPU (using OpenMP)" << std::endl;

std::chrono::system_clock::time_point start, end;

int N = _size;

int i,j,k;
float cij;

start = std::chrono::system_clock::now();

#pragma omp parallel for private (j, k, cij)
for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
cij = 0.0f;
for (k = 0; k < N; k++)
{
cij += _matrixA[i][k] * _matrixB[k][j];
}
_matrixC[i][j] = cij;
}
}

end = std::chrono::system_clock::now();
double elapsedTimeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
std::cout << "CheckResult: " << (CheckResult() ? "OK" : "NG") << std::endl;
std::cout << "----------" << std::endl;
}

void HpcSamples::Multiprocessor::RunOnCpu()
{
std::cout << std::endl;
std::cout << "----------" << std::endl;
std::cout << "Matrix Multiplication on CPU (Single Thread)" << std::endl;

std::chrono::system_clock::time_point start, end;

int N = _size;

int i,j,k;
float cij;

start = std::chrono::system_clock::now();

for (i = 0; i < N; i++)
{
for (j = 0; j < N; j++)
{
cij = 0.0f;
for (k = 0; k < N; k++)
{
cij += _matrixA[i][k] * _matrixB[k][j];
}
_matrixC[i][j] = cij;
}
}

end = std::chrono::system_clock::now();
double elapsedTimeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
std::cout << "CheckResult: " << (CheckResult() ? "OK" : "NG") << std::endl;
std::cout << "----------" << std::endl;
}
