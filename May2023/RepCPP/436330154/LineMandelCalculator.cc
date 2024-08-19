
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
data = (int*)(malloc(height * width * sizeof(int)));
r_array = (float*)(malloc(width * 2 * sizeof(float)));
i_array = (float*)(malloc(height * 2 * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
free(data);
free(r_array);
free(i_array);
data = NULL;
r_array = NULL;
i_array = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
int *pdata = data;
float *real_array = r_array;
float *imag_array = i_array;

for (int i = 0; i < sizeof(height * width * sizeof(int)); i++) {
pdata[i] = 0;
}

for (int i = 0; i < height; i++) {
float y = (float) y_start + i * (float) dy;

for (int k = 0; k < limit; ++k) {

#pragma omp simd 
for (int j = 0; j < width; j++) {
float x = (float) x_start + j * (float) dx;

float zReal = (k == 0) ? x : real_array[j];
float zImag = (k == 0) ? y : imag_array[j];

float r2 = zReal * zReal;
float i2 = zImag * zImag;

if (r2 + i2 < 4.0f) {
pdata[i * width + j] += 1;
real_array[j] = r2 - i2 + x;
imag_array[j] = 2.0f * zReal * zImag + y;
}
}
}
}
return data;
}
