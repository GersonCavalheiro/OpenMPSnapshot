#include <opencv2/opencv.hpp>

float lerp(float s, float e, float t) {
return s + (e - s) * t;
}

cv::Vec3f bInterp(cv::Vec3f c00, cv::Vec3f c10, cv::Vec3f c01, cv::Vec3f c11, float tx, float ty) {

float bchannel = lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty);
float gchannel = lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty);
float rchannel = lerp(lerp(c00[2], c10[2], tx), lerp(c01[2], c11[2], tx), ty);

return cv::Vec3f(bchannel, gchannel, rchannel);
}

void scalingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float scaling)
{
dst.create(scaling * src.rows, scaling * src.cols);
dst = cv::Vec3b(0, 0, 0);

float x_ratio = float(src.cols - 1) / dst.cols;
float y_ratio = float(src.rows - 1) / dst.rows;

#pragma omp parallel for
for (int row = 0; row < dst.rows; row++)
{
for (int col = 0; col < dst.cols; col++)
{
float gx, gy, tx, ty;

gx, tx = std::modf(row * x_ratio, &gx);
gy, ty = std::modf(col * y_ratio, &gy);
cv::Vec3f c00 = (cv::Vec3f)src.at<cv::Vec3b>(gx, gy);
cv::Vec3f c10 = (cv::Vec3f)src.at<cv::Vec3b>(gx + 1, gy);
cv::Vec3f c01 = (cv::Vec3f)src.at<cv::Vec3b>(gx, gy + 1);
cv::Vec3f c11 = (cv::Vec3f)src.at<cv::Vec3b>(gx + 1, gy + 1);
dst.at<cv::Vec3b>(row, col) = bInterp(c00, c10, c01,c11, tx, ty);
}
}
}

void denoisingOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int percent)
{

int k = (kernelSize - 1) / 2;

dst.create(src.rows - 2 * k, src.cols - 2 * k);
dst = cv::Vec3f(0, 0, 0);

#pragma omp parallel for
for (int i = k; i < src.rows - k; i++)
{
for (int j = k; j < src.cols - k; j++)
{
std::vector<float> rNeighbors;
std::vector<float> gNeighbors;
std::vector<float> bNeighbors;

float rValue = 0.0;
float gValue = 0.0;
float bValue = 0.0;

for (int u = i - k; u <= i + k; u++)
{
for (int v = j - k; v <= j + k; v++)
{
rNeighbors.push_back(src.at<cv::Vec3b>(u, v)[2]);
gNeighbors.push_back(src.at<cv::Vec3b>(u, v)[1]);
bNeighbors.push_back(src.at<cv::Vec3b>(u, v)[0]);
}
}

std::sort(rNeighbors.begin(), rNeighbors.end());
std::sort(gNeighbors.begin(), gNeighbors.end());
std::sort(bNeighbors.begin(), bNeighbors.end());

int medianIndx = 0;
if (kernelSize % 2 == 0) {
medianIndx = kernelSize * kernelSize / 2;
}
else {
medianIndx = (kernelSize * kernelSize - 1) / 2;
}

int numEl = (kernelSize * kernelSize * int(percent) / 100) / 2;

for (int w = (medianIndx - numEl); w <= (medianIndx + numEl); w++) {
rValue += rNeighbors[w];
gValue += gNeighbors[w];
bValue += bNeighbors[w];
}

if (numEl >= 1) {
rValue /= float(2 * numEl);
gValue /= float(2 * numEl);
bValue /= float(2 * numEl);
}

dst(i - k, j - k) = cv::Vec3f(bValue, gValue, rValue);

}
}
}

void gaussianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, int kernelSize, int sigma)
{

int k = (kernelSize - 1) / 2;

dst.create(src.rows - 2 * k, src.cols - 2 * k);
dst = cv::Vec3b(0, 0, 0);

#pragma omp parallel for
for (int i = k; i < src.rows - k; i++)
{
for (int j = k; j < src.cols - k; j++)
{
float tmp[3] = { 0.0, 0.0, 0.0, };
float kernelSum[3] = { 0.0, 0.0, 0.0, };
for (int u = i - k; u <= i + k; u++)
{
for (int v = j - k; v <= j + k; v++)
{
for (int c = 0; c < 3; c++)
{
tmp[c] += (float)src(u, v)[c] * exp(-((u - i) * (u - i) + (v - j) * (v - j)) / (2.0 * sigma * sigma));
kernelSum[c] += exp(-((u - i) * (u - i) + (v - j) * (v - j)) / (2.0 * sigma * sigma));
}
}
}

for (int c = 0; c < 3; c++)
{
dst(i - k, j - k)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

}
}
}
}

void laplacianConvOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst)
{

int k = 1;

dst.create(src.rows - 2 * k, src.cols - 2 * k);
dst = cv::Vec3b(0, 0, 0);

const int laplace_mat[3][3] = {
{-1, -1, -1},
{-1,  8, -1},
{-1, -1, -1}
};

#pragma omp parallel for
for (int i = k; i < src.rows - k; i++)
{
for (int j = k; j < src.cols - k; j++)
{
cv::Vec3f tmp = cv::Vec3f(0, 0, 0);
for (int u = i - k; u <= i + k; u++)
{
for (int v = j - k; v <= j + k; v++)
{
tmp += (cv::Vec3f)src.at<cv::Vec3b>(u, v) * laplace_mat[u - i + k][v - j + k];
}
}

dst.at<cv::Vec3b>(i - k, j - k) = tmp;
}
}
}

void gaussianSepOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& tmp_img, int kernelSize, int sigma)
{

int k = (kernelSize - 1) / 2;

dst.create(src.rows - 2 * k, src.cols - 2 * k);
dst = cv::Vec3b(0, 0, 0);

tmp_img.create(src.rows - 2 * k, src.cols);
tmp_img = cv::Vec3b(0, 0, 0);

#pragma omp parallel for
for (int i = k; i < src.rows - k; i++)
{
for (int j = 0; j < src.cols; j++)
{
float tmp[3] = { 0.0, 0.0, 0.0, };
float kernelSum[3] = { 0.0, 0.0, 0.0, };
for (int u = i - k; u <= i + k; u++)
{
for (int c = 0; c < 3; c++)
{
tmp[c] += (float)src(u, j)[c] * exp(-((u - i) * (u - i)) / (2.0 * sigma * sigma));
kernelSum[c] += exp(-((u - i) * (u - i)) / (2.0 * sigma * sigma));
}
}

for (int c = 0; c < 3; c++)
{
tmp_img(i - k, j)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

}
}
}

#pragma omp parallel for
for (int i = 0; i < tmp_img.rows; i++)
{
for (int j = k; j < tmp_img.cols - k; j++)
{
float tmp[3] = { 0.0, 0.0, 0.0, };
float kernelSum[3] = { 0.0, 0.0, 0.0, };
for (int u = j - k; u <= j + k; u++)
{
for (int c = 0; c < 3; c++)
{
tmp[c] += (float)tmp_img(i, u)[c] * exp(-((u - j) * (u - j)) / (2.0 * sigma * sigma));
kernelSum[c] += exp(-((u - j) * (u - j)) / (2.0 * sigma * sigma));
}
}

for (int c = 0; c < 3; c++)
{
dst(i, j - k)[c] = (unsigned char)(tmp[c] / kernelSum[c]);

}
}
}

}


cv::Vec3f bgr2xyz(const cv::Vec3f src) {

float scr_r = src[2] / 255.0;
float scr_g = src[1] / 255.0;
float scr_b = src[0] / 255.0;

cv::Vec3f tmp = cv::Vec3f(
(scr_r > .04045) ? pow((scr_r + .055) / 1.055, 2.4) : scr_r / 12.92,
(scr_g > .04045) ? pow((scr_g + .055) / 1.055, 2.4) : scr_g / 12.92,
(scr_b > .04045) ? pow((scr_b + .055) / 1.055, 2.4) : scr_b / 12.92
);

float mat[3][3] = {
.4124, .3576, .1805,
.2126, .7152, .0722,
.0193, .1192, .9505
};

tmp *= 100.0;

cv::Vec3f xyz;

for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
xyz[i] += mat[i][j] * tmp[j];
}
}

return xyz;
}
cv::Vec3f xyz2lab(const cv::Vec3f src) {

float scr_z = src[2] / 108.883;
float scr_y = src[1] / 100.;
float scr_x = src[0] / 95.047;

cv::Vec3f v = cv::Vec3f(
(scr_x > .008856) ? pow(scr_x, 1. / 3.) : (7.787 * scr_x) + (16. / 116.),
(scr_y > .008856) ? pow(scr_y, 1. / 3.) : (7.787 * scr_y) + (16. / 116.),
(scr_z > .008856) ? pow(scr_z, 1. / 3.) : (7.787 * scr_z) + (16. / 116.)
);

cv::Vec3f lab = cv::Vec3f((116. * v[1]) - 16., 500. * (v[0] - v[1]), 200. * (v[1] - v[2]));

return lab;
}

cv::Vec3f bgr2lab(cv::Vec3f c) {
return xyz2lab(bgr2xyz(c));
}

cv::Vec3f lab2xyz(cv::Vec3f src) {

float fy = (src[0] + 16.0) / 116.0;
float fx = src[1] / 500.0 + fy;
float fz = fy - src[2] / 200.0;

return cv::Vec3f(
95.047 * ((fx > 0.206897) ? fx * fx * fx : (fx - 16.0 / 116.0) / 7.787),
100.000 * ((fy > 0.206897) ? fy * fy * fy : (fy - 16.0 / 116.0) / 7.787),
108.883 * ((fz > 0.206897) ? fz * fz * fz : (fz - 16.0 / 116.0) / 7.787)
);
}
cv::Vec3f xyz2bgr(cv::Vec3f src) {

src /= 100.0;

float mat[3][3] = {
3.2406, -1.5372, -0.4986,
-0.9689, 1.8758, 0.0415,
0.0557, -0.2040, 1.0570
};

cv::Vec3f tmp;

for (int i = 0; i < 3; i++)
{
for (int j = 0; j < 3; j++)
{
tmp[i] += mat[i][j] * src[j];
}
}

cv::Vec3f bgr;
bgr[2] = (tmp[0] > 0.0031308) ? ((1.055 * pow(tmp[0], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[0]);
bgr[1] = (tmp[1] > 0.0031308) ? ((1.055 * pow(tmp[1], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[1]);
bgr[0] = (tmp[2] > 0.0031308) ? ((1.055 * pow(tmp[2], (1.0 / 2.4))) - 0.055) : 12.92 * (tmp[2]);

bgr *= 255.0;

return bgr;
}

cv::Vec3f lab2bgr(const cv::Vec3f src) {
return xyz2bgr(lab2xyz(src));
}

void colorTransfOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, float angle)
{

float PI = 3.14159265358979323846;

dst.create(src.rows, src.cols);
dst = cv::Vec3b(0, 0, 0);

cv::Mat_<cv::Vec3f> lab;
lab.create(src.rows, src.cols);
lab = cv::Vec3f(0, 0, 0);

#pragma omp parallel for
for (int i = 0; i < src.rows; i++)
{
for (int j = 0; j < src.cols; j++)
{
lab(i, j) = bgr2lab((cv::Vec3f)src.at<cv::Vec3b>(i, j));

float C = sqrt(lab(i, j)[1]* lab(i, j)[1] + lab(i, j)[2]* lab(i, j)[2]);
float h = atan2(lab(i, j)[2], lab(i, j)[1]);
h += (angle * PI) / 180.0;
lab(i, j)[1] = cos(h) * C;
lab(i, j)[2] = sin(h) * C;
dst(i, j) = lab2bgr((cv::Vec3f)lab.at<cv::Vec3f>(i, j));
}
}
}

void imageCombOpenmp(cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, cv::Mat_<cv::Vec3b>& src2, int imageComb, float offSet, float scaleFactor)
{

dst.create(src.rows, src.cols);
dst = cv::Vec3f(0, 0, 0);

#pragma omp parallel for
for (int i = 0; i < src.rows; i++)
{
for (int j = 0; j < src.cols; j++)
{
for (int c = 0; c < 3; c++)
{
if (imageComb == 0) {
dst(i, j)[c] = src(i, j)[c] + src2(i, j)[c];
}
else if (imageComb == 1) {
dst(i, j)[c] = src(i, j)[c] - src2(i, j)[c];
}
else if (imageComb == 2) {
dst(i, j)[c] = src(i, j)[c] * src2(i, j)[c];
}
else if (imageComb == 3) {
dst(i, j)[c] = src2(i, j)[c] == 0 ? src(i, j)[c] : (src(i, j)[c] / src2(i, j)[c]);
}
}

for (int c = 0; c < 3; c++)
{
dst(i, j)[c] *= scaleFactor;
dst(i, j)[c] += offSet;
}
}
}
}

