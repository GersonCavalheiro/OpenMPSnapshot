#include "opencv2/opencv.hpp"
#include "EmbossFilter.hpp"
#include <omp.h>

using namespace cv;



uchar convolutePixelEfficient(Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
int kSize = kernel.rows;
int halfSize = kSize / 2;

double newPixelValue = 0;

for (int i = 0; i < kSize; i++)
{
for (int j = 0; j < kSize; j++)
{
double inputImagePixelValue = static_cast<double>(inputImage.ptr<Vec3b>(I + i - halfSize, J + j - halfSize)->val[K]);
newPixelValue += inputImagePixelValue * kernel.at<float>(i, j);
}
}
return static_cast<uchar>(min(255, max(0, int(round(newPixelValue)))));
}

void applyEmbossFilterEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

outputImage = Mat::zeros(inputImage.size(), inputImage.type());

Vec3b *outputImagePointer;

for (int i = 1; i < inputImage.rows - 1; i++)
{
for (int j = 1; j < inputImage.cols - 1; j++)
{
outputImagePointer = outputImage.ptr<Vec3b>(i, j);

for (int k = 0; k < inputImage.channels(); k++)
{
outputImagePointer->val[k] = convolutePixelEfficient(inputImage, embossKernel, i, j, k);
}
}
}
}



void applyParallelEmbossFilter(Mat &inputImage, Mat &outputImage)
{
float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

outputImage = Mat::zeros(inputImage.size(), inputImage.type());

Vec3b *outputImagePointer;

#pragma omp parallel for private(outputImagePointer)
for (int i = 1; i < inputImage.rows - 1; i++)
{
for (int j = 1; j < inputImage.cols - 1; j++)
{
outputImagePointer = outputImage.ptr<Vec3b>(i, j);

for (int k = 0; k < inputImage.channels(); k++)
{
outputImagePointer->val[k] = convolutePixelEfficient(inputImage, embossKernel, i, j, k);
}
}
}
}



uchar convolutePixel(const Mat &inputImage, const Mat &kernel, int I, int J, int K)
{
int kSize = kernel.rows;
int halfSize = kSize / 2;

double pixelValue = 0;

for (int i = 0; i < kSize; i++)
{
for (int j = 0; j < kSize; j++)
{
uchar inputImagePixel = inputImage.at<Vec3b>(I + i - halfSize, J + j - halfSize)[K];
pixelValue += static_cast<double>(inputImagePixel) * kernel.at<float>(i, j);
}
}
return static_cast<uchar>(min(255, max(0, int(round(pixelValue)))));
}

void applyEmbossFilterSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
float emboss_data[9] = {2, -0, 0, 0, -1, 0, 0, 0, -1};
Mat embossKernel = Mat(3, 3, CV_32F, emboss_data);

outputImage = Mat::zeros(inputImage.size(), inputImage.type());

for (int i = 1; i < inputImage.rows - 1; i++)
{
for (int j = 1; j < inputImage.cols - 1; j++)
{
for (int k = 0; k < inputImage.channels(); k++)
{
outputImage.at<Vec3b>(i, j)[k] = convolutePixel(inputImage, embossKernel, i, j, k);
}
}
}
}
