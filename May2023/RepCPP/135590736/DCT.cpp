#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <omp.h>
#include "sys/time.h"
#include "DCT.h"

using namespace std;

#define MATRIX_SIZE 256
#define BLOCK_SIZE 8

int main() {
struct timeval start,end;
double elapsedTime;

const int32_t matrixSize = MATRIX_SIZE;

float *h_img = (float *)malloc(matrixSize * matrixSize * sizeof(float));
float *h_imgCopy = (float *)malloc(matrixSize * matrixSize * sizeof(float));
float *dctCoeffMatrix = (float *)malloc(matrixSize * matrixSize * sizeof(float));
float *temp = (float *)malloc(matrixSize * matrixSize * sizeof(float));

mt19937 rng(time(NULL));
uniform_int_distribution<int> gen(0, 255);

for (int i = 0; i < matrixSize; i++) {
for (int j = 0;j < matrixSize; j++) {
*(h_img + i * matrixSize + j) = static_cast<float>(gen(rng));
*(h_imgCopy + i * matrixSize + j) = *(h_img + i * matrixSize + j) - 128;
}
}

gettimeofday(&start, NULL);

#pragma omp parallel for collapse(2)
for (int i = 0; i < matrixSize; i += BLOCK_SIZE) {
for (int j = 0; j < matrixSize; j += BLOCK_SIZE) {


float sum = 0;

for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
for (int k = 0; k < BLOCK_SIZE; k++) {
sum += dct8x8Matrix[ii%BLOCK_SIZE][k] * *(h_imgCopy + k * matrixSize + jj);
}
*(temp + ii * matrixSize + jj) = sum;
sum = 0;
}
}


sum = 0;

for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
for (int k = 0; k < BLOCK_SIZE; k++) {
sum += *(temp + ii * matrixSize + k) * dct8x8MatrixTranspose[k][jj%BLOCK_SIZE];
}
*(dctCoeffMatrix + ii * matrixSize + jj) = sum;
sum = 0;
}
}
}
}

gettimeofday(&end, NULL);
elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
elapsedTime /= 1000.0;
printf("Multi Core CPU  : %.3f seconds\n", elapsedTime);
return 0;
}
