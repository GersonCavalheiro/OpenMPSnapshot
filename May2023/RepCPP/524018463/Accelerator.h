#pragma once

#include <cstdint>

namespace HpcSamples
{
class Accelerator
{
public:
Accelerator(uint16_t size);
~Accelerator();

void Initialize(uint16_t size);
bool CheckResult();
void RunOnAccelerator();
void RunOnCpu();

private:
uint16_t _size;
float** _matrixA;
float** _matrixB;
float** _matrixC;
};
}