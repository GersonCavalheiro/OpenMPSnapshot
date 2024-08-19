

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
data = (int*)(malloc(height * width * sizeof(int)));
r_array = (float*)(malloc(width * 2 * sizeof(float)));
i_array = (float*)(malloc(height * 2 * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
free(data);
free(r_array);
free(i_array);
data = NULL;
r_array = NULL;
i_array = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
int* pdata = data;
float* real_array = r_array;
float* imag_array = i_array;
const int blockSize = 64;

for (int i = 0; i < sizeof(height * width * sizeof(int)); i++) {
pdata[i] = 0;
}

for (int i = 0; i < height; i++) {
float y = (float)y_start + i * (float)dy;
int* ptr = &pdata[i * width];

for (int k = 0; k < limit; ++k) {

for (int block = 0; block < width / blockSize; block++) {

#pragma omp simd
for (int j = 0; j < blockSize; j++) {

int jGlobal = block * blockSize + j;

float x = (float)x_start + jGlobal * (float)dx;

float zReal = (k == 0) ? x : real_array[jGlobal];
float zImag = (k == 0) ? y : imag_array[jGlobal];

float r2 = zReal * zReal;
float i2 = zImag * zImag;

if (r2 + i2 < 4.0f) {
ptr[jGlobal] += 1;
real_array[jGlobal] = r2 - i2 + x;
imag_array[jGlobal] = 2.0f * zReal * zImag + y;
}
}
}
}
}
return data;
}