#include <stdio.h>
#include <string.h>
#include "sys/time.h"
#include "DCT.h"

#define MATRIX_SIZE 256
#define BLOCK_SIZE 8

float h_img[MATRIX_SIZE][MATRIX_SIZE],
h_imgCopy[MATRIX_SIZE][MATRIX_SIZE],
dctCoeffMatrix[MATRIX_SIZE][MATRIX_SIZE],
temp[MATRIX_SIZE][MATRIX_SIZE];

int main() {
struct timeval start,end;
double elapsedTime;

srand(time(0));

for (int i = 0; i < MATRIX_SIZE; i++) {
for (int j = 0;j < MATRIX_SIZE; j++) {
h_img[MATRIX_SIZE][MATRIX_SIZE] = (float)((rand() + 0) % (256));
h_imgCopy[i][j] = h_img[i][j] - 128;
}
}

gettimeofday(&start, NULL);

#pragma acc kernels
for (int i = 0; i < MATRIX_SIZE; i += BLOCK_SIZE) {
for (int j = 0; j < MATRIX_SIZE; j += BLOCK_SIZE) {


float sum = 0;

for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
for (int k = 0; k < BLOCK_SIZE; k++) {
sum += dct8x8Matrix[ii%BLOCK_SIZE][k] * h_imgCopy[k][jj];
}
temp[ii][jj] = sum;
sum = 0;
}
}


sum = 0;

for (int ii = i;ii < i+BLOCK_SIZE; ii++) {
for (int jj = j;jj < j+BLOCK_SIZE; jj++) {
for (int k = 0; k < BLOCK_SIZE; k++) {
sum += temp[ii][k] * dct8x8MatrixTranspose[k][jj%BLOCK_SIZE];
}
dctCoeffMatrix[ii][jj] = sum;
sum = 0;
}
}
}
}

gettimeofday(&end, NULL);
elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
elapsedTime /= 1000.0;
printf("OpenACC         : %.3f seconds\n", elapsedTime);

return 0;
}
