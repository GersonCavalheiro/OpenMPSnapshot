#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <omp.h>
#include "../include/config.hh"
#include "../../../include/stb_image.h"
#include "../../../include/stb_image_write.h"
#include "quantization.hh"
#include "dequantization.hh"
using namespace std;
using pixel_t = uint8_t;
int n, m;
void discreteCosTransform(int, int);
void free_mat(float **);
void divideMatrix(int, int);
inline int getOffset(int width, int i, int j) {
return (i * width + j) * 3;
}
vector<vector<int>> initializeIntMatrix(int rows, int cols) {
return vector<vector<int>>(rows, vector<int>(cols));
}
vector<vector<float>> initializeFloatMatrix(int rows, int cols) {
return vector<vector<float>>(rows, vector<float>(cols));
}
void divideMatrix(int n, int m) {
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < n; i += WINDOW_X) {
for (int j = 0; j < m; j += WINDOW_Y) {
discreteCosTransform(i, j);
}
}
}
void discreteCosTransform(int offsetX, int offsetY) {
int u, v, x, y;
float cos1, cos2, temp;
for (u = 0; u < WINDOW_X; ++u) {
for (v = 0; v < WINDOW_Y; ++v) {
temp = 0.0;
for (x = 0; x < WINDOW_X; x++) {
for (y = 0; y < WINDOW_Y; y++) {
cos1 = cosArr1[x][u];
cos2 = cosArr2[y][v];
temp += grayContent[x + offsetX][y + offsetY] * cos1 * cos2;
}
}
temp *= one_by_root_2N;
if (u > 0) {
temp *= one_by_root_2;
}
if (v > 0) {
temp *= one_by_root_2;
}
globalDCT[u + offsetX][v + offsetY] = (int)temp;
}
}
}
void compress(pixel_t *const img, int width, int height) {
n = height;
m = width;
int add_rows = (PIXEL - (n % PIXEL) != PIXEL ? PIXEL - (n % PIXEL) : 0);
int add_columns = (PIXEL - (m % PIXEL) != PIXEL ? PIXEL - (m % PIXEL) : 0) ;
int _height = grayContent.size();
int _width = grayContent[0].size();
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < n; i++) {
for(int j = 0; j < m; j++) {
pixel_t *bgrPixel = img + getOffset(width, i, j);
grayContent[i][j] = (bgrPixel[0] + bgrPixel[1] + bgrPixel[2]) / 3.f;
}
}
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int j = 0; j < m; j++) {
for (int i = n; i < n + add_rows; i++) {
grayContent[i][j] = 0;
}
}
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < n; i++) {
for (int j = m; j < m + add_columns; j++) {
grayContent[i][j] = 0;
}
}
n = _height;  
m = _width;   
#ifdef TIMER
auto start = chrono::high_resolution_clock::now();
divideMatrix(n, m);
auto end = chrono::high_resolution_clock::now();
std::chrono::duration<double> diff = end - start;
cout << "DCT: " << diff.count() << ", ";
start = chrono::high_resolution_clock::now();
quantize(n, m);
end = chrono::high_resolution_clock::now();
diff = end - start;
cout << "Quant: " << diff.count() << ", ";
start = chrono::high_resolution_clock::now();
dequantize(n, m);
end = chrono::high_resolution_clock::now();
diff = end - start;
cout << "Dequant: " << diff.count() << ", ";
start = chrono::high_resolution_clock::now();
invDct(n, m);
end = chrono::high_resolution_clock::now();
diff = end - start;
cout << "IDCT: " << diff.count() << ", ";
#else
divideMatrix(n, m);
quantize(n, m);
dequantize(n, m);
invDct(n, m);
#endif
#if !SERIAL
#ifdef OMP
#pragma omp parallel for schedule(runtime)
#endif
#endif
for (int i = 0; i < n; i++) {
for(int j = 0; j < m; j++) {
pixel_t pixelValue = finalMatrixDecompress[i][j];
pixel_t *bgrPixel = img + getOffset(width, i, j);
bgrPixel[0] = pixelValue;
bgrPixel[1] = pixelValue;
bgrPixel[2] = pixelValue;
}
}
}
int main(int argc, char **argv) {
FILE *fp;
fp = fopen("./info.txt","a+"); 
omp_set_num_threads(NUM_THREADS);
string img_dir = "../../../images/";
string save_dir = "./compressed_images/";
string ext = ".jpg";
string img_name = argv[1] + ext;
string path = img_dir + img_name;
cout << img_name << ", ";
#ifdef OMP
cout << "OMP, ";
#endif
#ifdef SIMD
cout << "SIMD, ";
#endif
#if SERIAL
cout << "SERIAL, ";
#endif
cosArr1 = vector<vector<float>>(8, vector<float>(8));
cosArr2 = vector<vector<float>>(8, vector<float>(8));
for (int i = 0; i < 8; i++) {
for (int j = 0; j < 8; j++) {
cosArr1[i][j] = cos(term1 * (i + 0.5) * j);
cosArr2[i][j] = cos(term2 * (i + 0.5) * j);
}
}
int width, height, bpp;
pixel_t *const img = stbi_load(path.data(), &width, &height, &bpp, 3);
cout << "Width: " << width << ", ";
cout << "Height: " << height << ", ";
int add_rows = (PIXEL - (height % PIXEL) != PIXEL ? PIXEL - (height % PIXEL) : 0);
int add_columns = (PIXEL - (width % PIXEL) != PIXEL ? PIXEL - (width % PIXEL) : 0) ;
int _height = height + add_rows;
int _width = width + add_columns;
grayContent = initializeIntMatrix(_height, _width);
globalDCT = initializeFloatMatrix(_height, _width);
finalMatrixCompress = initializeIntMatrix(_height, _width);
finalMatrixDecompress = initializeIntMatrix(_height, _width);
auto start = chrono::high_resolution_clock::now();
compress(img, width, height);
auto end = chrono::high_resolution_clock::now();
std::chrono::duration<double> diff_parallel = end - start;
#if SERIAL
string save_img = save_dir + "ser_" + img_name;
stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#else
string save_img = save_dir + "par_" + img_name;
stbi_write_jpg(save_img.data(), width, height, bpp, img, width * bpp);
#endif
stbi_image_free(img);
#if SERIAL
cout << "Serial -> ";
#else
cout << "Parallel -> ";
#endif   
cout << diff_parallel.count() << endl;
fprintf(fp,"%f ",(float)diff_parallel.count());
fclose(fp);
return 0;
}
