#pragma once

class CudaFilterer {
public:
float** gaussian_pyramid; 

float* cudaImageData;
float* cudaGaussianPyramid;	
int imageWidth;
int imageHeight;
int numLevels;

CudaFilterer();
virtual ~CudaFilterer();

void setup(float* img, int h, int w);

void allocHostGaussianPyramid(int width, int height, int num_levels);
void allocDeviceGaussianPyramid(int width, int height);

float** createGaussianPyramid(float sigma0, float k, const int* levels, 
int num_levels);

void getGaussianPyramid(int i);
};