#pragma once

#include <cstdint>

namespace HpcSamples
{
class Multiprocessor
{
public:
Multiprocessor(uint16_t n);
~Multiprocessor();

void Initialize(uint16_t size);
bool CheckResult();
void RunOnMultiprocessor();
void RunOnCpu();

private:
uint16_t _size;
float** _matrixA;
float** _matrixB;
float** _matrixC;
};
}