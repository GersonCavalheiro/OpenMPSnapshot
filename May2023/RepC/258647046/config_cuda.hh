#pragma once
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define PIXEL 8
#define WINDOW_X 8
#define WINDOW_Y 8
#define SERIAL 0
#if !SERIAL
#define CUDA
#endif
#ifdef CUDA
#define BLK_WIDTH 8
#define BLK_HEIGHT 8
#define BLOCKSIZE (BLK_HEIGHT * BLK_WIDTH)
#endif
#define TIMER
#define NUM_CHANNELS 3
extern uint8_t *cudaImg;
void cudaSetup(uint8_t *img, int width, int height);
void compress(int width, int height);
void cudaFinish(uint8_t *img, int width, int height);
const float term1 = M_PI / (float)WINDOW_X;
const float term2 = M_PI / (float)WINDOW_Y;
const float term3 = 2. / (float)WINDOW_X;
const float term4 = 2. / (float)WINDOW_Y;
const float one_by_root_2 = 1.0 / sqrt(2);
const float one_by_root_2N = 1.0 / sqrt(2 * WINDOW_X);