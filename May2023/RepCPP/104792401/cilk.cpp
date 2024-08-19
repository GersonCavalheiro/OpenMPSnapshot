#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

using namespace cv;

struct pixel {
double red;
double green;
double blue;

pixel(double r, double g, double b) : red(r), green(g), blue(b) {};
};


void prewittX_kernel(const int rows, const int cols, double * const kernel) {
if(rows != 3 || cols !=3) {
std::cerr << "Bad Prewitt kernel matrix\n";
return;
}
for(int i=0;i<3;i++) {
kernel[0 + (i*rows)] = -1.0;
kernel[1 + (i*rows)] = 0.0;
kernel[2 + (i*rows)] = 1.0;
}
}

void prewittY_kernel(const int rows, const int cols, double * const kernel) {
if(rows != 3 || cols !=3) {
std::cerr << "Bad Prewitt kernel matrix\n";
return;
}
for(int i=0;i<3;i++) {
kernel[i + (0*rows)] = 1.0;
kernel[i + (1*rows)] = 0.0;
kernel[i + (2*rows)] = -1.0;
}
}


void gaussian_kernel(const int rows, const int cols, const double stddev, double * const kernel) {
const double denom = 2.0 * stddev * stddev;
const double g_denom = M_PI * denom;
const double g_denom_recip = (1.0/g_denom);
double sum = 0.0;

for(int i = 0; i < rows; ++i) {
for(int j = 0; j < cols; ++j) {
const double row_dist = i - (rows/2);
const double col_dist = j - (cols/2);
const double dist_sq = (row_dist * row_dist) + (col_dist * col_dist);
const double value = g_denom_recip * exp((-dist_sq)/denom);
kernel[i + (j*rows)] = value;
sum += value;
}
}

const double recip_sum = 1.0 / sum;
for(int i = 0; i < rows; ++i) {
for(int j = 0; j < cols; ++j) {
kernel[i + (j*rows)] *= recip_sum;
}
}
}


void apply_stencil(const int radius, const double stddev, const int rows, const int cols, pixel * const in, pixel * const out) {
const int dim = radius*2+1;
double kernel[dim*dim];
gaussian_kernel(dim, dim, stddev, kernel);
int pre_n = 3;
double prewittX[pre_n*pre_n];
double prewittY[pre_n*pre_n];
prewittX_kernel(pre_n,pre_n,prewittX);
prewittY_kernel(pre_n,pre_n,prewittY);
double* intensity = (double*) malloc(sizeof(double)*rows*cols);
cilk_for(int i = 0; i < rows; ++i) {
cilk_for(int j = 0; j < cols; ++j) {
const int out_offset = i + (j*rows);
double red = 0, green = 0, blue = 0;
for(int x = i - radius; x <= i + radius; ++x) {
for(int y = j - radius; y <= j + radius; ++y) {
if(x >= 0 && x < rows && y >= 0 && y < cols) {
int kxx = x - (i - radius);
int kyy = y - (j - radius);
const int in_offset = x + (y*rows);
const int k_offset = kxx + (kyy*dim);
red   += kernel[k_offset] * in[in_offset].red;
green += kernel[k_offset] * in[in_offset].green;
blue  += kernel[k_offset] * in[in_offset].blue;
}
}
}
out[out_offset].red = red;
out[out_offset].green = green;
out[out_offset].blue = blue;
intensity[out_offset] = (out[out_offset].red + out[out_offset].green + out[out_offset].blue)/3.0;
}
}
cilk_for(int i = 0; i < rows; ++i){
cilk_for(int j = 0; j < cols; ++j){
int out_offset = i + (j*rows);
double prewitt_x = 0, prewitt_y = 0;
for (int x = i-1; x<=i+1; x++){
for (int y = j-1; y<=j+1; y++){
if(x >= 0 && x < rows && y >= 0 && y < cols){
int kxx = x - (i - 1);
int kyy = y - (j - 1);
const int in_offset = x + (y*rows);
const int k_offset= kxx + (kyy*3);
prewitt_x+=prewittX[k_offset]*intensity[in_offset];
prewitt_y+=prewittY[k_offset]*intensity[in_offset];
}
}
}
out[out_offset].red = out[out_offset].green = out[out_offset].blue = sqrt(prewitt_x * prewitt_x + prewitt_y * prewitt_y);
}
}
}

int main( int argc, char* argv[] ) {

if(argc != 2) {
std::cerr << "Usage: " << argv[0] << " imageName\n";
return 1;
}

Mat image;
image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
if(!image.data ) {
std::cout <<  "Error opening " << argv[1] << std::endl;
return -1;
}

const int rows = image.rows;
const int cols = image.cols;
pixel * imagePixels = (pixel *) malloc(rows * cols * sizeof(pixel));
for(int i = 0; i < rows; ++i) {
for(int j = 0; j < cols; ++j) {
Vec3b p = image.at<Vec3b>(i, j);
imagePixels[i + (j*rows)] = pixel(p[0]/255.0,p[1]/255.0,p[2]/255.0);
}
}

pixel * outPixels = (pixel *) malloc(rows * cols * sizeof(pixel));
for(int i = 0; i < rows * cols; ++i) {
outPixels[i].red = 0.0;
outPixels[i].green = 0.0;
outPixels[i].blue = 0.0;
}

struct timespec start_time;
struct timespec end_time;
clock_gettime(CLOCK_MONOTONIC,&start_time);
apply_stencil(5, 32.0, rows, cols, imagePixels, outPixels);
clock_gettime(CLOCK_MONOTONIC,&end_time);
long msec = (end_time.tv_sec - start_time.tv_sec)*1000 + (end_time.tv_nsec - start_time.tv_nsec)/1000000;
printf("Stencil application took %dms\n",msec);

Mat dest(rows, cols, CV_8UC3);
for(int i = 0; i < rows; ++i) {
for(int j = 0; j < cols; ++j) {
const size_t offset = i + (j*rows);
dest.at<Vec3b>(i, j) = Vec3b(floor(outPixels[offset].red * 255.0),
floor(outPixels[offset].green * 255.0),
floor(outPixels[offset].blue * 255.0));
}
}

imwrite("out.jpg", dest);


free(imagePixels);
free(outPixels);
return 0;
}
