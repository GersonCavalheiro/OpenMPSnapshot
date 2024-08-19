#include "opencv2/opencv.hpp"
#include "RgbToGrayscale.hpp"
#include <omp.h>

using namespace cv;




void RgbToGrayscaleEfficientPixelAccess(Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

if (outputImage.channels() > 1)
{
throw("Image has to much channels for Grayscale!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

int grayPixelValue;

Vec3b *inputImagePointer;
uchar *outputImagePointer;

for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
inputImagePointer = inputImage.ptr<Vec3b>(i, j);
outputImagePointer = outputImage.ptr<uchar>(i, j);

double R = static_cast<double>(inputImagePointer->val[2]);
double G = static_cast<double>(inputImagePointer->val[1]);
double B = static_cast<double>(inputImagePointer->val[0]);

grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

*outputImagePointer = grayPixelValue;
}
}
}



void RgbToGrayscaleParallel(Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

if (outputImage.channels() > 1)
{
throw("Image has to much channels for Grayscale!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

int grayPixelValue;

Vec3b *inputImagePointer;
uchar *outputImagePointer;

#pragma omp parallel for private(inputImagePointer, outputImagePointer, grayPixelValue)
for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
inputImagePointer = inputImage.ptr<Vec3b>(i, j);
outputImagePointer = outputImage.ptr<uchar>(i, j);

double R = static_cast<double>(inputImagePointer->val[2]);
double G = static_cast<double>(inputImagePointer->val[1]);
double B = static_cast<double>(inputImagePointer->val[0]);

grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

*outputImagePointer = grayPixelValue;
}
}
}



void RgbToGrayscaleSlowPixelAccess(const Mat &inputImage, Mat &outputImage)
{
if (inputImage.channels() < 3)
{
throw("Image doesn't have enough channels for RGB!");
}

if (outputImage.channels() > 1)
{
throw("Image has to much channels for Grayscale!");
}

outputImage = Mat::zeros(inputImage.size(), CV_8UC1);

Vec3b bgr_pixel;
int grayPixelValue;

for (int i = 0; i < inputImage.rows; i++)
{
for (int j = 0; j < inputImage.cols; j++)
{
bgr_pixel = inputImage.at<Vec3b>(i, j);

double R = static_cast<double>(bgr_pixel[2]);
double G = static_cast<double>(bgr_pixel[1]);
double B = static_cast<double>(bgr_pixel[0]);

grayPixelValue = 0.21 * R + 0.72 * G + 0.07 * B;

outputImage.at<uchar>(i, j) = grayPixelValue;
}
}
}
