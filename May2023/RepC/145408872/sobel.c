#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#pragma omp declare target
#include <math.h>
#include "sobel.h"
#include "macros.h"
int rgbToGray(byte * __restrict__ rgb, byte * __restrict__ gray, int buffer_size) {
int gray_size = buffer_size / 3;
#pragma omp parallel for
for(int i=0; i<gray_size; i++)
gray[i] = 0.30*rgb[i*3] + 0.59*rgb[i*3+1] + 0.11*rgb[i*3+2];
return gray_size;
}
void makeOpMem(byte *buffer, int buffer_size, int width, int cindex, byte *op_mem) {
int bottom = cindex-width < 0;
int top = cindex+width >= buffer_size;
int left = cindex % width == 0;
int right = (cindex+1) % width == 0;
op_mem[0] = !bottom && !left  ? buffer[cindex-width-1] : 0;
op_mem[1] = !bottom           ? buffer[cindex-width]   : 0;
op_mem[2] = !bottom && !right ? buffer[cindex-width+1] : 0;
op_mem[3] = !left             ? buffer[cindex-1]       : 0;
op_mem[4] = buffer[cindex];
op_mem[5] = !right            ? buffer[cindex+1]       : 0;
op_mem[6] = !top && !left     ? buffer[cindex+width-1] : 0;
op_mem[7] = !top              ? buffer[cindex+width]   : 0;
op_mem[8] = !top && !right    ? buffer[cindex+width+1] : 0;
}
int convolution(byte * __restrict__ X, int * __restrict__ Y, int c_size) {
int sum = 0;
for(int i=0; i<c_size; i++) {
sum += X[i] * Y[c_size-i-1];
}
return sum;
}
void itConv(byte *buffer, int buffer_size, int width, int *op, byte * __restrict__ res) {
byte op_mem[SOBEL_OP_SIZE];
memset(op_mem, 0, SOBEL_OP_SIZE);
#pragma omp parallel for firstprivate(op_mem)
for(int i=0; i<buffer_size; i++) {
makeOpMem(buffer, buffer_size, width, i, op_mem);
res[i] = (byte) abs(convolution(op_mem, op, SOBEL_OP_SIZE));
}
}
void contour(byte * __restrict__ sobel_h, byte * __restrict__ sobel_v, int gray_size, byte * __restrict__ contour_img) {
#pragma omp parallel for
for(int i=0; i<gray_size; i++) {
contour_img[i] = (byte) sqrt(pow(sobel_h[i], 2) + pow(sobel_v[i], 2));
}
}
int sobelFilter(byte *rgb, byte *gray, byte *sobel_h_res, byte *sobel_v_res, byte *contour_img, int width, int height) {
int sobel_h[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1},
sobel_v[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
int rgb_size = width*height*3;
int gray_size = rgbToGray(rgb, gray, rgb_size);
itConv(gray, gray_size, width, sobel_h, sobel_h_res);
itConv(gray, gray_size, width, sobel_v, sobel_v_res);
contour(sobel_h_res, sobel_v_res, gray_size, contour_img);
return gray_size;
}
#pragma omp end declare target
