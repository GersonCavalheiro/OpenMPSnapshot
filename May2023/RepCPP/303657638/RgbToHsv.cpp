#include "opencv2/opencv.hpp"
#include "RgbToHsv.hpp"
#include <omp.h>

using namespace cv;




void RgbToHsvEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

Vec3b *inputImagePointer;
Vec3b *outputImagePointer;

for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
inputImagePointer = inputImage.ptr<Vec3b>(i, j);
outputImagePointer = outputImage.ptr<Vec3b>(i, j);

double R = static_cast<double>(inputImagePointer->val[2]) / 255.0;
double G = static_cast<double>(inputImagePointer->val[1]) / 255.0;
double B = static_cast<double>(inputImagePointer->val[0]) / 255.0;

double cmin = min({R, G, B});
double cmax = max({R, G, B});
double diff = cmax - cmin;

if (cmin == cmax)
{
outputImagePointer->val[0] = 0;
}

if (cmax == R)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
}
else if (cmax == G)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
}
else if (cmax == B)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
}

if (cmax == 0)
{
outputImagePointer->val[1] = 0;
}
else
{
outputImagePointer->val[1] = static_cast<uchar>((diff / cmax) * 100);
}

outputImagePointer->val[2] = static_cast<uchar>(cmax * 255.0);
}
}
}



void RgbToHsvParallel(Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

Vec3b *inputImagePointer;
Vec3b *outputImagePointer;

#pragma omp parallel for private(inputImagePointer, outputImagePointer)
for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
inputImagePointer = inputImage.ptr<Vec3b>(i, j);
outputImagePointer = outputImage.ptr<Vec3b>(i, j);

double R = static_cast<double>(inputImagePointer->val[2]) / 255.0;
double G = static_cast<double>(inputImagePointer->val[1]) / 255.0;
double B = static_cast<double>(inputImagePointer->val[0]) / 255.0;

double cmin = min({R, G, B});
double cmax = max({R, G, B});
double diff = cmax - cmin;

if (cmin == cmax)
{
outputImagePointer->val[0] = 0;
}

if (cmax == R)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
}
else if (cmax == G)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
}
else if (cmax == B)
{
outputImagePointer->val[0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
}

if (cmax == 0)
{
outputImagePointer->val[1] = 0;
}
else
{
outputImagePointer->val[1] = static_cast<uchar>((diff / cmax) * 100);
}

outputImagePointer->val[2] = static_cast<uchar>(cmax * 255.0);
}
}
}



void RgbToHsvSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC3);

Vec3b bgr_pixel;
Vec3b hsv_pixel;

for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
bgr_pixel = inputImage.at<Vec3b>(i, j);

double R = static_cast<double>(bgr_pixel[2]) / 255.0;
double G = static_cast<double>(bgr_pixel[1]) / 255.0;
double B = static_cast<double>(bgr_pixel[0]) / 255.0;

double cmin = min({R, G, B});
double cmax = max({R, G, B});
double diff = cmax - cmin;

if (cmin == cmax)
{
hsv_pixel[0] = 0;
}

if (cmax == R)
{
hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((G - B) / diff) + 360.0) % 360);
}
else if (cmax == G)
{
hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((B - R) / diff) + 120.0) % 360);
}
else if (cmax == B)
{
hsv_pixel[0] = static_cast<uchar>(static_cast<int>(60 * ((R - G) / diff) + 240.0) % 360);
}

if (cmax == 0)
{
hsv_pixel[1] = 0;
}
else
{
hsv_pixel[1] = static_cast<uchar>((diff / cmax) * 100);
}

hsv_pixel[2] = static_cast<uchar>(cmax * 255.0);

outputImage.at<Vec3b>(i, j) = hsv_pixel;
}
}
}
