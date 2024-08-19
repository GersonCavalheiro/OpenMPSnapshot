#include "stencil.h"

using namespace std;

template<typename P>
void ApplyStencil(ImageClass<P> & img_in, ImageClass<P> & img_out) {

const int width  = img_in.width;
const int height = img_in.height;

P * in  = img_in.pixel;
P * out = img_out.pixel;

for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
P val = -in[(i-1)*width + j-1] -   in[(i-1)*width + j] - in[(i-1)*width + j+1] 
-in[(i  )*width + j-1] + 8*in[(i  )*width + j] - in[(i  )*width + j+1] 
-in[(i+1)*width + j-1] -   in[(i+1)*width + j] - in[(i+1)*width + j+1];

val = (val < 0   ? 0   : val);
val = (val > 255 ? 255 : val);

out[i*width + j] = val;
}

}

template<typename P>
void Convolve(ImageClass<P> & img_in, ImageClass<P> & img_out, int** kernel) {
const int width  = img_in.width;
const int height = img_in.height;

P * in  = img_in.pixel;
P * out = img_out.pixel;

for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
P val = kernel[0][0]*in[(i-1)*width + j-1] +
kernel[0][1]*in[(i-1)*width + j]   +
kernel[0][2]*in[(i-1)*width + j+1] +
kernel[1][0]*in[(i  )*width + j-1] +
kernel[1][1]*in[(i  )*width + j]   +
kernel[1][2]*in[(i  )*width + j+1] +
kernel[2][0]*in[(i+1)*width + j-1] +
kernel[2][1]*in[(i+1)*width + j]   +
kernel[2][2]*in[(i+1)*width + j+1];

out[i*width + j] = val;
}
}

int** simple_kernel() {
int** array2D = 0;
array2D = new int*[3];
for (int h = 0; h < 3; h++) {
array2D[h] = new int[3];
for (int w = 0; w < 3; w++) {
array2D[h][w] = 1;
}
}
return array2D;
}

void sobel_filter(ImageClass<float> &img_in, ImageClass<float> &img_out, 
ImageClass<float> &_theta) {
int** K = simple_kernel();
K[0] = new int[3] {-1, 0, 1};
K[1] = new int[3] {-2, 0, 2};
K[2] = new int[3] {-1, 0, 1};
Convolve(img_in, img_out, K);

const int width = img_in.width;
const int height = img_in.height;
float* Ix = (float*)_mm_malloc(sizeof(float)*width*height, 64);
#pragma omp parallel for
for (int i = 0; i < height; i++)
for (int j = 0; j < width; j++)
Ix[i*width + j] = img_out.pixel[i*width + j];

K[0] = new int[3] {1, 2, 1};
K[1] = new int[3] {0, 0, 0};
K[2] = new int[3] {-1, -2, -1};
Convolve(img_in, img_out, K);

float* gradient = img_out.pixel;
float* atan_vals = _theta.pixel;
for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
atan_vals[i*width+j] = atan2(gradient[i*width+j], Ix[i*width+j]) * 180/PI;
if (atan_vals[i*width+j] < 0) atan_vals[i*width+j] += 180;
}

for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
gradient[i*width + j] = sqrt(pow(Ix[i*width+j], 2.0)+pow(gradient[i*width+j], 2.0));
}
}

void Convolve_mpi(ImageClass<float> & img_in, ImageClass<float> & img_out,
int** kernel, int first_row, int last_row) {
const int width  = img_in.width;
const int height = img_in.height;

float * in  = img_in.pixel;
float * out = img_out.pixel;

for (int i = first_row; i < last_row; i++)
for (int j = 1; j < width-1; j++) {
float val = kernel[0][0]*in[(i-1)*width + j-1] +
kernel[0][1]*in[(i-1)*width + j]   +
kernel[0][2]*in[(i-1)*width + j+1] +
kernel[1][0]*in[(i  )*width + j-1] +
kernel[1][1]*in[(i  )*width + j]   +
kernel[1][2]*in[(i  )*width + j+1] +
kernel[2][0]*in[(i+1)*width + j-1] +
kernel[2][1]*in[(i+1)*width + j]   +
kernel[2][2]*in[(i+1)*width + j+1];

out[i*width + j] = val;
}
}


void sobel_filter_mpi(ImageClass<float> &img_in, ImageClass<float> &img_out, 
ImageClass<float> &_theta, const int myFirstRow, const int myLastRow) {
const int width = img_in.width;
const int height = img_in.height;
int** K = simple_kernel();
K[0] = new int[3] {-1, 0, 1};
K[1] = new int[3] {-2, 0, 2};
K[2] = new int[3] {-1, 0, 1};
Convolve_mpi(img_in, img_out, K, myFirstRow, myLastRow);

float* Ix = (float*)_mm_malloc(sizeof(float)*width*height, 64);
#pragma omp parallel for
for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 0; j < width; j++)
Ix[i*width + j] = img_out.pixel[i*width + j];

K[0] = new int[3] {1, 2, 1};
K[1] = new int[3] {0, 0, 0};
K[2] = new int[3] {-1, -2, -1};
Convolve_mpi(img_in, img_out, K, myFirstRow, myLastRow);

float* gradient = img_out.pixel;
float* atan_vals = _theta.pixel;
for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 1; j < width-1; j++) {
atan_vals[i*width+j] = atan2(gradient[i*width+j], Ix[i*width+j]) * 180/PI;
if (atan_vals[i*width+j] < 0) atan_vals[i*width+j] += 180;
}

for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 1; j < width-1; j++) {
gradient[i*width + j] = sqrt(pow(Ix[i*width+j], 2.0)+pow(gradient[i*width+j], 2.0));
}
}

void non_max_suppress(ImageClass<float> &img_grad, ImageClass<float> &img_theta) {
const int width  = img_grad.width;
const int height = img_grad.height;
float* gradient = img_grad.pixel;
float* _theta = img_theta.pixel;
ImageClass<float> res(width, height);
float* res_vals = res.pixel;

for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
int q = 255;
int r = 255;
if ((0 <= _theta[i*width+j] < 22.5) || (157.5 <= _theta[i*width+j] <= 180)) {
q = gradient[i*width+j+1];
r = gradient[i*width+j-1];
} else if (22.5 <= _theta[i*width+j] < 67.5) {
q = gradient[(i+1)*width+j-1];
r = gradient[(i-1)*width+j+1];
} else if (67.5 <= _theta[i*width+j] < 112.5) {
q = gradient[(i+1)*width+j];
r = gradient[(i-1)*width+j];
} else if (112.5 <= _theta[i*width+j] < 157.5) {
q = gradient[(i-1)*width+j-1];
r = gradient[(i+1)*width+j+1];
} else;

if ((gradient[i*width+j] >= q) && (gradient[i*width+j]) >= r) res_vals[i*width+j] = gradient[i*width+j];
else res_vals[i*width+j] = 0;
}

for (int i = 0; i < height-1; i++)
for (int j = 0; j < width-1; j++) {
gradient[i*width+j] = res_vals[i*width+j];
} 
}

void non_max_suppress_mpi(ImageClass<float> &img_grad, ImageClass<float> &img_theta, const int myFirstRow, const int myLastRow) {
const int width  = img_grad.width;
const int height = img_grad.height;
float* gradient = img_grad.pixel;
float* _theta = img_theta.pixel;
ImageClass<float> res(width, height);
float* res_vals = res.pixel;

for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 1; j < width-1; j++) {
int q = 255;
int r = 255;
if ((0 <= _theta[i*width+j] < 22.5) || (157.5 <= _theta[i*width+j] <= 180)) {
q = gradient[i*width+j+1];
r = gradient[i*width+j-1];
} else if (22.5 <= _theta[i*width+j] < 67.5) {
q = gradient[(i+1)*width+j-1];
r = gradient[(i-1)*width+j+1];
} else if (67.5 <= _theta[i*width+j] < 112.5) {
q = gradient[(i+1)*width+j];
r = gradient[(i-1)*width+j];
} else if (112.5 <= _theta[i*width+j] < 157.5) {
q = gradient[(i-1)*width+j-1];
r = gradient[(i+1)*width+j+1];
} else;

if ((gradient[i*width+j] >= q) && (gradient[i*width+j]) >= r) res_vals[i*width+j] = gradient[i*width+j];
else res_vals[i*width+j] = 0;
}

for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 0; j < width-1; j++) {
gradient[i*width+j] = res_vals[i*width+j];
}
}

void threshold(ImageClass<float> &img) {
float max_val = 0;
const int width  = img.width;
const int height = img.height;
float* pixels = img.pixel;
for (int i = 0; i < height-1; i++)
for (int j = 0; j < width-1; j++) {
if (pixels[i*width+j] > max_val) max_val = pixels[i*width+j];
} 
float high_threshold = max_val*0.15;
float low_threshold = high_threshold * 0.05;
for (int i = 0; i < height-1; i++)
for (int j = 0; j < width-1; j++) {
if (pixels[i*width+j] > high_threshold) pixels[i*width+j] = 255;
else if (low_threshold <= pixels[i*width+j] && pixels[i*width+j] <= high_threshold) {
pixels[i*width+j] = 75;
} else pixels[i*width+j] = 0;
}
}

void threshold_mpi(ImageClass<float> &img, const int myFirstRow, const int myLastRow, float max_val) {
const int width  = img.width;
const int height = img.height;
float* pixels = img.pixel;
for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 0; j < width-1; j++) {
if (pixels[i*width+j] > max_val) max_val = pixels[i*width+j];
} 
float high_threshold = max_val*0.15;
float low_threshold = high_threshold * 0.05;
for (int i = myFirstRow; i < myLastRow; i++)
for (int j = 0; j < width-1; j++) {
if (pixels[i*width+j] > high_threshold) pixels[i*width+j] = 255;
else if (low_threshold <= pixels[i*width+j] && pixels[i*width+j] <= high_threshold) {
pixels[i*width+j] = 75;
} else pixels[i*width+j] = 0;
}
}

void tracking(ImageClass<float> &img) {
const int width  = img.width;
const int height = img.height;
float* pixels = img.pixel;
for (int i = 1; i < height-1; i++)
for (int j = 1; j < width-1; j++) {
if (pixels[i*width+j] == 75) {
if (check_blob(pixels, i, j, width)) {
pixels[i*width+j] = 255;
} else pixels[i*width+j] = 0;
}
}
}

bool check_blob(float* &pixels_arr, int x, int y, int width) {
bool res;  
if (pixels_arr[(x-1)*width+y-1] == 255 ||
pixels_arr[(x-1)*width+y]   == 255 ||
pixels_arr[(x-1)*width+y+1] == 255 ||
pixels_arr[(x)*width+y-1]   == 255 ||
pixels_arr[(x)*width+y+1]   == 255 ||
pixels_arr[(x+1)*width+y-1] == 255 || 
pixels_arr[(x+1)*width+y]   == 255 ||
pixels_arr[(x+1)*width+y+1] == 255) {
res = true;
} else res = false;
return res;
}

void tracking_mpi(ImageClass<float> &img, const int myFirstRow, const int myLastRow, int rank, bool first_pass) {
const int width  = img.width;
const int height = img.height;
float* pixels = img.pixel;
bool bool2 = true;
for (int u=0; u<width; u++) {
if (pixels[(510)*width+u] != 0) {
bool2 = false;
break;
}
}
if (first_pass) {
for (int i = myFirstRow+1; i < myLastRow-1; i++) {
for (int j = 1; j < width-1; j++) {
if (pixels[i*width+j] == 75) {
if (check_blob(pixels, i, j, width)) {
pixels[i*width+j] = 255;
} else pixels[i*width+j] = 0;
}
}
}
} else {
for (int j = 1; j < width-1; j++) {
if (pixels[myFirstRow*width+j] == 75) {
if (check_blob(pixels, myFirstRow, j, width)) {
pixels[myFirstRow*width+j] = 255;
} else pixels[myFirstRow*width+j] = 0;
}
if (pixels[(myLastRow-1)*width+j] == 75) {
if (check_blob(pixels, myLastRow-1, j, width)) {
pixels[(myLastRow-1)*width+j] = 255;
} else pixels[(myLastRow-1)*width+j] = 0;
}
}
} 
}

template void Convolve(ImageClass<float> & img_in, ImageClass<float> & img_out, int** kernel);
template void ApplyStencil<float>(ImageClass<float> & img_in, ImageClass<float> & img_out);
