#pragma once
#include <vector>
#include <cmath>
#include "../include/config.hh"
#include "quantization.hh"
using namespace std;
void invDiscreteCosTransform(int R, int C) {
int x, y, u, v;
float cos1, cos2, temp;
for (u = 0; u < WINDOW_X; ++u) {
for (v = 0; v < WINDOW_Y; ++v) {
temp = 1/4. * (float)finalMatrixCompress[R + 0][C + 0];
for (x = 1; x < WINDOW_X; x++) {
temp += 1/2. * (float)finalMatrixCompress[R + x][C + 0];
}
for (y = 1; y < WINDOW_Y; y++) {
temp += 1/2. * (float)finalMatrixCompress[R + 0][C + y];
}
for (x = 1; x < WINDOW_X; x++) {
for (y = 1; y < WINDOW_Y; y++) {
cos1 = cosArr1[x][u];
cos2 = cosArr2[y][v];
temp += (float)finalMatrixCompress[R + x][C + y] * cos1 * cos2;
}
}
finalMatrixDecompress[u + R][v + C] = temp * term3 * term4;
}
}
}
void invDct(int height, int width) {
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < height; i += WINDOW_X) {
for (int j = 0; j < width; j += WINDOW_Y) {
invDiscreteCosTransform(i, j);
}
}
}
void dequantizeBlock(int R, int C) {
int i, j;
for (i = 0; i < WINDOW_X; i++) {
for (j = 0; j < WINDOW_Y; j++) {
finalMatrixCompress[R + i][C + j] *= quantArr[i][j];
}
}
}
void dequantize(int height, int width) {
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < height; i += WINDOW_X) {
for (int j = 0; j < width; j += WINDOW_Y) {
dequantizeBlock(i, j);
}
}
}
