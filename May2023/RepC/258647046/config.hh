#pragma once
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define PIXEL 8
#define WINDOW_X 8
#define WINDOW_Y 8
#define SERIAL 0
#if !SERIAL
#define OMP
#endif
#define TIMER
#define NUM_THREADS 8
#define NUM_CHANNELS 3
std::vector<std::vector<int>> grayContent;
std::vector<std::vector<float>> globalDCT;
std::vector<std::vector<int>> finalMatrixCompress;
std::vector<std::vector<int>> finalMatrixDecompress;
std::vector<std::vector<float>> cosArr1;
std::vector<std::vector<float>> cosArr2;
const float one_by_root_2 = 1.0 / sqrt(2);
const float one_by_root_2N = 1.0 / sqrt(2 * WINDOW_X);
const float term1 = M_PI / (float)WINDOW_X;
const float term2 = M_PI / (float)WINDOW_Y;
const float term3 = 2. / (float)WINDOW_X;
const float term4 = 2. / (float)WINDOW_Y;