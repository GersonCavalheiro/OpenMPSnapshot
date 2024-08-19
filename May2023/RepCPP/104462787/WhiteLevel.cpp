#include "WhiteLevel.h"


namespace Capture3
{

WhiteLevel::WhiteLevel(const Image *image) :
width(image->getSize().getWidth()),
height(image->getSize().getHeight()),
area(image->getSize().getArea()),
white(image->getRGB().getMat().clone()),
whiteData((double *) white.data),

whiteLevelMinR(0),
whiteLevelMinG(0),
whiteLevelMinB(0),
whiteLevelMaxR(0),
whiteLevelMaxG(0),
whiteLevelMaxB(0)
{
cv::blur(white, white, cv::Size(50, 50));
cv::blur(white, white, cv::Size(50, 50));
cv::blur(white, white, cv::Size(50, 50));

double minR = 1;
double minG = 1;
double minB = 1;
double maxR = 0;
double maxG = 0;
double maxB = 0;

#pragma omp parallel for schedule(static) \
reduction(min:minR), \
reduction(min:minG), \
reduction(min:minB), \
reduction(max:maxR), \
reduction(max:maxG), \
reduction(max:maxB)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;
const double r = whiteData[index + 0];
const double g = whiteData[index + 1];
const double b = whiteData[index + 2];

minR = r < minR ? r : minR;
minG = g < minG ? g : minG;
minB = b < minB ? b : minB;
maxR = r > maxR ? r : maxR;
maxG = g > maxG ? g : maxG;
maxB = b > maxB ? b : maxB;
}

whiteLevelMinR = minR;
whiteLevelMinG = minG;
whiteLevelMinB = minB;
whiteLevelMaxR = maxR;
whiteLevelMaxG = maxG;
whiteLevelMaxB = maxB;

whiteLevelMin = std::min(std::min(whiteLevelMinR, whiteLevelMinG), whiteLevelMinB);
whiteLevelMax = std::max(std::max(whiteLevelMaxR, whiteLevelMaxG), whiteLevelMinB);

whiteLevelR = whiteLevelMax / whiteLevelMaxR;
whiteLevelG = whiteLevelMax / whiteLevelMaxG;
whiteLevelB = whiteLevelMax / whiteLevelMaxB;

whiteLevel = std::max(std::max(whiteLevelR, whiteLevelG), whiteLevelB);
whiteLevelNormR = whiteLevelR / whiteLevel;
whiteLevelNormG = whiteLevelG / whiteLevel;
whiteLevelNormB = whiteLevelB / whiteLevel;
}


WhiteLevel::~WhiteLevel()
{
whiteData = nullptr;
white.release();
}


void WhiteLevel::apply(Image *image)
{
double *imageData = image->getRGB().getData();

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < area; i++) {

const unsigned int index = i * 3;
double r = imageData[index + 0];
double g = imageData[index + 1];
double b = imageData[index + 2];

const double whiteR = whiteData[index + 0];
const double whiteG = whiteData[index + 1];
const double whiteB = whiteData[index + 2];

r = r * (whiteLevelMaxR / whiteR);
g = g * (whiteLevelMaxG / whiteG);
b = b * (whiteLevelMaxB / whiteB);
r = r * whiteLevelNormR;
g = g * whiteLevelNormG;
b = b * whiteLevelNormB;


imageData[index + 0] = r;
imageData[index + 1] = g;
imageData[index + 2] = b;
}

image->convertRGB();
}
}