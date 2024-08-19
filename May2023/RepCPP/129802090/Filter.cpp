
#include <cstring>
#include <cmath>
#include "Filter.h"
#include "Bitmap.h"
#include "Kernels.h"

Filter::Filter() {}
Filter::~Filter() {}

void Filter::Negative(Bitmap &image)
{
unsigned int width = image.GetWidth();
unsigned int height = image.GetHeight();

int x, y;
#pragma omp parallel for schedule(dynamic) private(x, y)
for (x = 0; x < width; x++)
{
for (y = 0; y < height; y++)
{
Color color = image.GetPixel(x, y);
unsigned char r, g, b;

r = 255 - color.red();
g = 255 - color.green();
b = 255 - color.blue();

image.SetPixel(x, y, Color(r, g, b));
}
}
}

void Filter::Grayscale(Bitmap &image)
{
unsigned int width = image.GetWidth();
unsigned int height = image.GetHeight();

int x, y;
#pragma omp parallel
{
#pragma omp for schedule(dynamic) private(x, y)
for (x = 0; x < width; x++)
{
for (y = 0; y < height; y++)
{
Color color = image.GetPixel(x, y);

unsigned char grayed = 0;
grayed += color.red() * 0.114;
grayed += color.green() * 0.587;
grayed += color.blue() * 0.299;

image.SetPixel(x, y, Color(grayed, grayed, grayed));
}
}
}
}

void ConvolutionFilter(Bitmap &sourceImage, const double xkernel[][3], const double ykernel[][3], double factor, int bias, bool grayscale)
{
int width = sourceImage.GetWidth();
int height = sourceImage.GetHeight();


int srcDataStride = sourceImage.GetStride();
int bytes = srcDataStride * sourceImage.GetHeight();

unsigned char* pixelBuffer = new unsigned char[bytes];
unsigned char* resultBuffer = new unsigned char[bytes];

memcpy(pixelBuffer, sourceImage.GetPixelData(), bytes);
int stepBytes = 3;

if (grayscale == true)
{
float rgb = 0;
for (int i = 0; i < sourceImage.GetPixelArraySize(); i += stepBytes)
{
rgb = pixelBuffer[i] * .21f;
rgb += pixelBuffer[i + 1] * .71f;
rgb += pixelBuffer[i + 2] * .071f;
pixelBuffer[i] = rgb;
pixelBuffer[i + 1] = pixelBuffer[i];
pixelBuffer[i + 2] = pixelBuffer[i];
}
}

double xr = 0.0;
double xg = 0.0;
double xb = 0.0;
double yr = 0.0;
double yg = 0.0;
double yb = 0.0;
double rt = 0.0;
double gt = 0.0;
double bt = 0.0;

int filterOffset = 1;
int calcOffset = 0;
int byteOffset = 0;

int OffsetY, OffsetX;

for (OffsetY = filterOffset; OffsetY < height - filterOffset; OffsetY++)
{
for (OffsetX = filterOffset; OffsetX < width - filterOffset; OffsetX++)
{
xr = xg = xb = yr = yg = yb = 0;
rt = gt = bt = 0.0;

byteOffset = OffsetY * srcDataStride + OffsetX * stepBytes;

for (int filterY = -filterOffset; filterY <= filterOffset; filterY++)
{
for (int filterX = -filterOffset; filterX <= filterOffset; filterX++)
{
calcOffset = byteOffset + filterX * 4 + filterY * srcDataStride;

xb += (double)(pixelBuffer[calcOffset])     * xkernel[filterY + filterOffset][filterX + filterOffset];
xg += (double)(pixelBuffer[calcOffset + 1]) * xkernel[filterY + filterOffset][filterX + filterOffset];
xr += (double)(pixelBuffer[calcOffset + 2]) * xkernel[filterY + filterOffset][filterX + filterOffset];

yb += (double)(pixelBuffer[calcOffset])     * ykernel[filterY + filterOffset][filterX + filterOffset];
yg += (double)(pixelBuffer[calcOffset + 1]) * ykernel[filterY + filterOffset][filterX + filterOffset];
yr += (double)(pixelBuffer[calcOffset + 2]) * ykernel[filterY + filterOffset][filterX + filterOffset];
}
}

bt = sqrt((xb * xb) + (yb * yb));
gt = sqrt((xg * xg) + (yg * yg));
rt = sqrt((xr * xr) + (yr * yr));

if (bt > 255) bt = 255;
else if (bt < 0) bt = 0;
if (gt > 255) gt = 255;
else if (gt < 0) gt = 0;
if (rt > 255) rt = 255;
else if (rt < 0) rt = 0;

resultBuffer[byteOffset] = (bt);
resultBuffer[byteOffset + 1] = (gt);
resultBuffer[byteOffset + 2] = (rt);
}
}
sourceImage.SetPixelData(resultBuffer);
}

void ConvolutionFilter(Bitmap &sourceImage, const double filterMatrix[][5], double factor, int bias, bool grayscale)
{
int width = sourceImage.GetWidth();
int height = sourceImage.GetHeight();

int srcDataStride = sourceImage.GetStride();
int bytes = srcDataStride * sourceImage.GetHeight();

unsigned char* pixelBuffer = new unsigned char[bytes];
unsigned char* resultBuffer = new unsigned char[bytes];

std::memcpy(pixelBuffer, sourceImage.GetPixelData(), bytes);
int stepBytes = 3;

if (grayscale == true)
{
float rgb = 0;
for (int i = 0; i < sourceImage.GetPixelArraySize(); i += stepBytes)
{
rgb = pixelBuffer[i] * .21f;
rgb += pixelBuffer[i + 1] * .71f;
rgb += pixelBuffer[i + 2] * .071f;
pixelBuffer[i] = rgb;
pixelBuffer[i + 1] = pixelBuffer[i];
pixelBuffer[i + 2] = pixelBuffer[i];
}
}

double red, green, blue;

int filterWidth = 5;
int filterHeight = 5;

int filterOffset = (filterWidth - 1) / 2;
int calcOffset = 0;

int byteOffset = 0;

for (int offsetY = filterOffset; offsetY < height - filterOffset; offsetY++)
{
for (int offsetX = filterOffset; offsetX < width - filterOffset; offsetX++)
{
blue = 0;
green = 0;
red = 0;

byteOffset = offsetY * srcDataStride + offsetX * stepBytes;

for (int filterY = -filterOffset; filterY <= filterOffset; filterY++)
{
for (int filterX = -filterOffset; filterX <= filterOffset; filterX++)
{
calcOffset = byteOffset + (filterX * 4) + (filterY * srcDataStride);

blue += (double)(pixelBuffer[calcOffset]) * filterMatrix[filterY + filterOffset][filterX + filterOffset];
green += (double)(pixelBuffer[calcOffset + 1]) * filterMatrix[filterY + filterOffset][filterX + filterOffset];
red += (double)(pixelBuffer[calcOffset + 2]) * filterMatrix[filterY + filterOffset][filterX + filterOffset];
}
}

blue = factor * blue + bias;
green = factor * green + bias;
red = factor * red + bias;

if (blue > 255) blue = 255;
else if (blue < 0) blue = 0;

if (green > 255) green = 255;
else if (green < 0) green = 0;

if (red > 255) red = 255;
else if (red < 0) red = 0;

resultBuffer[byteOffset] = (blue);
resultBuffer[byteOffset + 1] = (green);
resultBuffer[byteOffset + 2] = (red);
}
}

sourceImage.SetPixelData(resultBuffer);
}

void Filter::Blur(Bitmap & image)
{
ConvolutionFilter(image, gaussian5x5, 1 / 256.f, 0, false);
}

void Filter::Convolve(Bitmap & image, const double xkernel[][3], const double ykernel[][3], double factor, int bias, bool grayscale)
{
ConvolutionFilter(image, xkernel, ykernel, 1, 0, grayscale);
}

void Filter::Convolve(Bitmap & image, const double kernel[][5], double factor, int bias, bool grayscale)
{
ConvolutionFilter(image, kernel, factor, bias, grayscale);
}