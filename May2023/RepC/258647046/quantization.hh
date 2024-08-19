#pragma once
#include <vector>
#include <cmath>
#include "../include/config.hh"
using namespace std;
vector<vector<int>> quantArr = {{16, 11, 12, 14, 12, 10, 16, 14},
{13, 14, 18, 17, 16, 19, 24, 40},
{26, 24, 22, 22, 24, 49, 35, 37},
{29, 40, 58, 51, 61, 60, 57, 51},
{56, 55, 64, 72, 92, 78, 64, 68},
{87, 69, 55, 56, 80, 109, 81, 87},
{95, 98, 103, 104, 103, 62, 77, 113},
{121, 112, 100, 120, 92, 101, 103, 99}
};
void quantizeBlock(int R, int C) {
int i, j, temp;
for (i = 0; i < WINDOW_X; i++) {
for (j = 0; j < WINDOW_Y; j++) {
temp = globalDCT[R + i][C + j];
temp = (int)round((float)temp / quantArr[i][j]);
finalMatrixCompress[R + i][C + j] = temp;
}
}
}
void quantize(int height, int width) {
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < height; i += WINDOW_X) {
for (int j = 0; j < width; j += WINDOW_Y) {
quantizeBlock(i, j);
}
}
}
