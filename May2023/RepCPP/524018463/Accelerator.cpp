#include "Accelerator.h"
#include <openacc.h>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>

HpcSamples::Accelerator::Accelerator(uint16_t size)
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

HpcSamples::Accelerator::~Accelerator()
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

void HpcSamples::Accelerator::Initialize(uint16_t size)
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

bool HpcSamples::Accelerator::CheckResult()
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

void HpcSamples::Accelerator::RunOnAccelerator()
{
std::cout << std::endl;
std::cout << "----------" << std::endl;
std::cout << "Matrix Multiplication on Accelerator (using OpenACC)" << std::endl;

std::chrono::system_clock::time_point start, end;

int N = _size;
start = std::chrono::system_clock::now();

#pragma acc enter data copyin(this)
#pragma acc data copyout(_matrixC[0:N][0:N]) copyin(_matrixB[0:N][0:N], _matrixA[0:N][0:N])
#pragma acc kernels
#pragma acc loop independent gang
for (int i = 0; i < N; i++)
{
#pragma acc loop independent vector
for (int j = 0; j < N; j++)
{
float cij = 0.0f;
#pragma acc loop reduction(+:cij)
for (int k = 0; k < N; k++)
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

void HpcSamples::Accelerator::RunOnCpu()
{
std::cout << std::endl;
std::cout << "----------" << std::endl;
std::cout << "Matrix Multiplication on CPU (Single Thread)" << std::endl;

std::chrono::system_clock::time_point start, end;

int N = _size;
start = std::chrono::system_clock::now();

for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
{
float cij = 0.0f;
for (int k = 0; k < N; k++)
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
