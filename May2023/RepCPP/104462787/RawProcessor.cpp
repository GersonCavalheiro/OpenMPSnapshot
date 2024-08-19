#include "RawProcessor.h"


#include <QDebug>


namespace Capture3
{

RawProcessor::RawProcessor() :
files(),

outputWidth(CAMERA_RAW_WIDTH),
outputHeight(CAMERA_RAW_HEIGHT),
outputArea(outputWidth * outputHeight),
outputSize(outputWidth, outputHeight)
{
}


RawProcessor::~RawProcessor()
{
files.clear();
}


void RawProcessor::add(
const unsigned int indexFocus,
const unsigned int indexExposure,
const unsigned int indexShot,
RawFile *file)
{
if (files.size() <= indexFocus) {
files.resize(indexFocus + 1);
}
if (files[indexFocus].size() <= indexExposure) {
files[indexFocus].resize(indexExposure + 1);
}
if (files[indexFocus][indexExposure].size() <= indexShot) {
files[indexFocus][indexExposure].resize(indexShot + 1);
}

files[indexFocus][indexExposure][indexShot] = file;
}


Image *RawProcessor::process(
const unsigned int mergeShots,
const unsigned int mergeRange
)
{
outputWidth = CAMERA_RAW_WIDTH;
outputHeight = CAMERA_RAW_HEIGHT;
outputArea = outputWidth * outputHeight;
outputSize.width = outputWidth;
outputSize.height = outputHeight;

std::vector<cv::Mat> images;

openImages(images, mergeShots, mergeRange);
alignImages(images);
return stackImages(images);
}


void RawProcessor::openImages(
std::vector<cv::Mat> &images,
const unsigned int mergeShots,
const unsigned int mergeRange
)
{
std::vector<cv::Mat> imagesRange;
std::vector<cv::Mat> imagesShots;

for (unsigned int indexFocus = 0; indexFocus < files.size(); indexFocus++) {
for (unsigned int indexExposure = 0; indexExposure < files[indexFocus].size(); indexExposure++) {
for (unsigned int indexShot = 0; indexShot < files[indexFocus][indexExposure].size(); indexShot++) {

RawFile *file = files[indexFocus][indexExposure][indexShot];

if (processor.open_file(file->getFilePath().toLatin1().constData()) == LIBRAW_SUCCESS) {
processor.unpack();

cv::Mat image(outputSize, CV_64FC3, cv::Scalar(0));
auto *imageData = (double *) image.data;

const unsigned short *raw = processor.imgdata.rawdata.raw_image;
const unsigned int inputWidth = processor.imgdata.sizes.width;
const unsigned int inputHeight = processor.imgdata.sizes.height;
const unsigned int rangeMin = processor.imgdata.color.black;
const unsigned int rangeMax = processor.imgdata.color.maximum;
const unsigned int range = rangeMax - rangeMin;

#pragma omp parallel for schedule(static) collapse(2)
for (unsigned int y = 0; y < outputHeight; y++) {
for (unsigned int x = 0; x < outputWidth; x++) {

const unsigned int col = x * 2;
const unsigned int row = y * 2;

const unsigned int indexR = (inputWidth * row) + col;
const unsigned int indexG1 = (inputWidth * row) + col + 1;
const unsigned int indexG2 = (inputWidth * (row + 1)) + col;
const unsigned int indexB = (inputWidth * (row + 1)) + col + 1;

const unsigned int index = (y * outputWidth + x) * 3;

auto colorR = (double) raw[indexR];
auto colorG1 = (double) raw[indexG1];
auto colorG2 = (double) raw[indexG2];
auto colorB = (double) raw[indexB];
double colorG = (colorG1 + colorG2) / 2.0;

colorR = (colorR - rangeMin) / range;
colorG = (colorG - rangeMin) / range;
colorB = (colorB - rangeMin) / range;

colorR = colorR < 0 ? 0 : colorR > 1 ? 1 : colorR;
colorG = colorG < 0 ? 0 : colorG > 1 ? 1 : colorG;
colorB = colorB < 0 ? 0 : colorB > 1 ? 1 : colorB;

imageData[index + 0] = colorR;
imageData[index + 1] = colorG;
imageData[index + 2] = colorB;
}
}

imagesShots.push_back(image);
}

processor.free_image();
processor.recycle();
}

if (!imagesShots.empty()) {
imagesRange.push_back(
mergeImages(imagesShots, mergeShots)
);
imagesShots.clear();
}
}

if (!imagesRange.empty()) {
images.push_back(
mergeImages(imagesRange, mergeRange)
);
imagesRange.clear();
}
}

files.clear();
}


cv::Mat RawProcessor::mergeImages(std::vector<cv::Mat> &images, const unsigned int type)
{
const auto count = (unsigned int) images.size();

if (count == 1) {
return images[0].clone();
}

cv::Mat output(outputSize, CV_64FC3, cv::Scalar(0));
auto *outputData = (double *) output.data;

if (type != 2) {

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < outputArea * 3; i++) {

std::vector<double> pixels(count, 0.0);
for (unsigned int n = 0; n < count; n++) {
pixels[n] = ((double *) images[n].data)[i];
}

if (type == 0) {
outputData[i] = calculateMean(pixels);
}
if (type == 1) {
outputData[i] = calculateMedian(pixels);
}
}

} else {

std::vector<cv::Mat> weights(count);
cv::Mat sum(outputSize, CV_64FC1, cv::Scalar(0));

const double contrastMultiplier = 1.0;
const double saturationMultiplier = 1.0;
const double exposureMultiplier = 1.0;

for (unsigned int i = 0; i < count; i++) {

cv::Mat &image = images[i];

cv::Mat greyscale(outputSize, CV_64FC1, cv::Scalar(0));
toGreyscale(
(double *) image.data,
(double *) greyscale.data,
outputArea
);


cv::Mat gradientX(outputSize, CV_64FC1, cv::Scalar(0));
cv::Mat gradientY(outputSize, CV_64FC1, cv::Scalar(0));
cv::Sobel(greyscale, gradientX, CV_64FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
cv::Sobel(greyscale, gradientY, CV_64FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
gradientX = cv::abs(gradientX);
gradientY = cv::abs(gradientY);
cv::Mat contrast(outputSize, CV_64FC1, cv::Scalar(0));
cv::addWeighted(gradientX, 0.5, gradientY, 0.5, 0, contrast, CV_64FC1);

std::vector<cv::Mat> channels(3);
cv::split(image, channels);

cv::Mat mean(outputSize, CV_64FC1, cv::Scalar(0));
mean += channels[0];
mean += channels[1];
mean += channels[2];
mean /= 3.0;

cv::Mat saturation(outputSize, CV_64FC1, cv::Scalar(0));
for (unsigned int c = 0; c < 3; c++) {
cv::Mat deviation = channels[c] - mean;
cv::pow(deviation, 2.0, deviation);
saturation += deviation;
}
cv::sqrt(saturation, saturation);

cv::Mat exposure(outputSize, CV_64FC1, double(1));
for (unsigned int c = 0; c < 3; c++) {
cv::Mat expo = channels[c] - 0.5;
cv::pow(expo, 2.0, expo);
expo = -expo / 0.08; 
cv::exp(expo, expo);
exposure = exposure.mul(expo);
}

cv::pow(contrast, contrastMultiplier, contrast);
cv::pow(saturation, saturationMultiplier, saturation);
cv::pow(exposure, exposureMultiplier, exposure);


cv::Mat weight(outputSize, CV_64FC1, double(1));
weight = weight.mul(contrast);
weight = weight.mul(saturation);
weight = weight.mul(exposure);


weight += 1e-12;
weights[i] = weight;
sum += weight;
}


auto maxLevel = static_cast<int>(std::log(std::min(outputWidth, outputHeight)) / std::log(2.0));

std::vector<cv::Mat> res_pyr((size_t) maxLevel + 1);

for (unsigned int i = 0; i < count; i++) {

cv::Mat &image = images[i];
cv::Mat &weight = weights[i];

weight /= sum;


std::vector<cv::Mat> imagePyramid;
std::vector<cv::Mat> weightPyramid;
cv::buildPyramid(image, imagePyramid, maxLevel);
cv::buildPyramid(weight, weightPyramid, maxLevel);

for (int level = 0; level < maxLevel; level++) {
cv::Mat up;
cv::pyrUp(imagePyramid[level + 1], up, imagePyramid[level].size());
imagePyramid[level] -= up;
}

for (int level = 0; level <= maxLevel; level++) {
std::vector<cv::Mat> channels(3);
cv::split(imagePyramid[level], channels);
channels[0] = channels[0].mul(weightPyramid[level]);
channels[1] = channels[1].mul(weightPyramid[level]);
channels[2] = channels[2].mul(weightPyramid[level]);

cv::merge(channels, imagePyramid[level]);
if (res_pyr[level].empty()) {
res_pyr[level] = imagePyramid[level];
} else {
res_pyr[level] += imagePyramid[level];
}
}
}

for (int level = maxLevel; level > 0; level--) {
cv::Mat up;
cv::pyrUp(res_pyr[level], up, res_pyr[level - 1].size());
res_pyr[level - 1] += up;
}


res_pyr[0].copyTo(output);
}

return output;
}


void RawProcessor::alignImages(std::vector<cv::Mat> &images)
{
const auto count = (unsigned int) images.size();

if (count >= 2) {

std::vector<cv::Mat> greyscales;

for (unsigned int i = 0; i < count; i++) {

const cv::Mat &image = images[i];

cv::Mat greyscale(outputSize, CV_64FC1, cv::Scalar(0));
toGreyscale(
(double *) image.data,
(double *) greyscale.data,
outputArea
);

cv::GaussianBlur(greyscale, greyscale, cv::Size(5, 5), 0, 0);

greyscales.push_back(greyscale);
}




double scaleTotal = 0;
double scaleMean = 0;

#pragma omp parallel for schedule(static)
for (unsigned int i = 0; i < count - 1; i++) {

const cv::Mat &imageNear = greyscales[i];
const cv::Mat &imageFar = greyscales[i + 1];

unsigned int searchStart = 0;
unsigned int searchEnd = 3;

unsigned int matchIndex = 0;
unsigned int matchStep = 0;
unsigned int matchWidth = outputWidth;
unsigned int matchHeight = outputHeight;
unsigned int matchX = 0;
unsigned int matchY = 0;
double matchScale = 1;

QHash<unsigned int, double> cache;

for (unsigned int power = 4; power >= 1; power--) {

const auto stepPixels = (unsigned int) std::pow(2, power);
const auto stepScaled = (double) stepPixels / outputWidth;

double prevError = 1;

for (unsigned int n = searchStart; n < searchEnd; n++) {

const unsigned int step = stepPixels * n;

const double scale = 1.0 - (stepScaled * n);

const auto regionWidth = (unsigned int) lround(outputWidth * scale);
const auto regionHeight = (unsigned int) lround(outputHeight * scale);
const auto regionX = (unsigned int) lround((outputWidth - regionWidth) / 2.0);
const auto regionY = (unsigned int) lround((outputHeight - regionHeight) / 2.0);
const cv::Rect regionRect(regionX, regionY, regionWidth, regionHeight);
const cv::Size regionSize(regionWidth, regionHeight);

if (!cache.contains(step)) {

cv::Mat region(imageFar, regionRect);

cv::Mat temp;
cv::resize(imageNear, temp, regionSize, scale, scale, cv::INTER_LINEAR);

cv::absdiff(region, temp, temp);

cache[step] = cv::mean(temp)[0];
}

const double error = cache[step];

if (error < prevError) {
matchIndex = n;
matchStep = step;
matchWidth = regionWidth;
matchHeight = regionHeight;
matchX = regionX;
matchY = regionY;
matchScale = scale;
}
prevError = error;

if (error == 0) {
break;
}
}

if (power == 1) {

#pragma omp atomic update
scaleTotal += matchScale;
break;
}

const int stepMatched = stepPixels * matchIndex;
const auto stepNext = (int) std::pow(2, power - 1);
const int stepStart = (stepMatched - stepNext) / stepNext;
searchStart = (unsigned int) (stepStart < 0 ? 0 : stepStart);
searchEnd = (unsigned int) (stepStart + 3);
}
}

scaleMean = scaleTotal / (count - 1);
scaleTotal = std::pow(scaleMean, (count - 1));

outputWidth = (unsigned int) lround(outputWidth * scaleTotal);
outputHeight = (unsigned int) lround(outputHeight * scaleTotal);
outputArea = outputWidth * outputHeight;
outputSize.width = outputWidth;
outputSize.height = outputHeight;

#pragma omp parallel for schedule(static)
for (unsigned int i = 1; i < count; i++) {

const double scale = std::pow(scaleMean, i);
const auto scaleWidth = (unsigned int) lround(outputWidth * scale);
const auto scaleHeight = (unsigned int) lround(outputHeight * scale);
const cv::Size scaleSize(scaleWidth, scaleHeight);

const auto cropX = (unsigned int) lround((scaleWidth - outputWidth) / 2.0);
const auto cropY = (unsigned int) lround((scaleHeight - outputHeight) / 2.0);
const cv::Rect cropRect(cropX, cropY, outputWidth, outputHeight);

cv::Mat &image = images[count - i - 1];

cv::resize(image, image, scaleSize, scale, scale, cv::INTER_AREA);
cv::Mat cropped(image, cropRect);
cropped.copyTo(image);
cropped.release();
}
}
}



Image *RawProcessor::stackImages(std::vector<cv::Mat> &images)
{
auto *output = new Image(outputWidth, outputHeight);
double *outputData = output->getRGB().getData();

const auto count = (unsigned int) images.size();

for (unsigned int i = 0; i < count; i++) {

const cv::Mat &image = images[i];
const double *imageData = (double *) image.data;

#pragma omp parallel for schedule(static)
for (unsigned int n = 0; n < outputArea; n++) {
const unsigned int index = n * 3;
const double colorR = imageData[index + 0] / count;
const double colorG = imageData[index + 1] / count;
const double colorB = imageData[index + 2] / count;
outputData[index + 0] += colorR;
outputData[index + 1] += colorG;
outputData[index + 2] += colorB;
}
}

if (count > 1) {

std::vector<cv::Mat> gradients;

for (unsigned int i = 0; i < count; i++) {

const cv::Mat &image = images[i];

cv::Mat greyscale(outputSize, CV_64FC1, cv::Scalar(0));
toGreyscale(
(double *) image.data,
(double *) greyscale.data,
outputArea
);

cv::Mat gradientX(outputSize, CV_64FC1, cv::Scalar(0));
cv::Mat gradientY(outputSize, CV_64FC1, cv::Scalar(0));
cv::Sobel(greyscale, gradientX, CV_64FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
cv::Sobel(greyscale, gradientY, CV_64FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
gradientX = cv::abs(gradientX);
gradientY = cv::abs(gradientY);
cv::Mat gradient(outputSize, CV_64FC1, cv::Scalar(0));
cv::addWeighted(gradientX, 0.5, gradientY, 0.5, 0, gradient, CV_64FC1);

greyscale.release();
gradientX.release();
gradientY.release();

gradients.push_back(gradient);
}



}

output->convertRGB();
return output;
}
}