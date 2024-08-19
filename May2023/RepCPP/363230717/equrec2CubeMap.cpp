#include <chrono> 
#include <iostream>
#include <math.h> 
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;

float getTheta(float x, float y) {
float rtn = 0;
if (y < 0) {
rtn = atan2(y, x) * -1;
} else {
rtn = M_PI + (M_PI - atan2(y, x));
}
return rtn;
}

int main(int argc, char **argv) {

cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);

cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

cv::imshow("Original Image", source);

const int inputWidth = source.cols;
const int inputHeight = source.rows;
const int sqr = source.cols / 4.0;
const int output_width = sqr * 3;
const int output_height = sqr * 2;

cv::Mat destination(output_height, output_width, CV_8UC3,
cv::Scalar(255, 255, 255));

auto begin = chrono::high_resolution_clock::now();

#pragma omp parallel for
for (int j = 0; j < destination.cols; j++) {
for (int i = 0; i < destination.rows; i++) {
float tx = 0.0;
float ty = 0.0;
float x = 0.0;
float y = 0.0;
float z = 0.0;

if (i < sqr + 1) {   
if (j < sqr + 1) { 
tx = j;
ty = i;
x = tx - 0.5 * sqr;
y = 0.5 * sqr;
z = ty - 0.5 * sqr;
} else if (j < 2 * sqr + 1) { 
tx = j - sqr;
ty = i;
x = 0.5 * sqr;
y = (tx - 0.5 * sqr) * -1;
z = ty - 0.5 * sqr;
}

else { 
tx = j - sqr * 2;
ty = i;
x = (tx - 0.5 * sqr) * -1;
y = -0.5 * sqr;
z = ty - 0.5 * sqr;
}
} else {             
if (j < sqr + 1) { 

tx = j;
ty = i - sqr;
x = int(-0.5 * sqr);
y = int(tx - 0.5 * sqr);
z = int(ty - 0.5 * sqr);
}

else if (j < 2 * sqr + 1) { 

tx = j - sqr;
ty = i - sqr;
x = (ty - 0.5 * sqr) * -1;
y = (tx - 0.5 * sqr) * -1;
z = 0.5 * sqr; 
}

else { 

tx = j - sqr * 2;
ty = i - sqr;
x = ty - 0.5 * sqr;
y = (tx - 0.5 * sqr) * -1;
z = -0.5 * sqr; 
}
}

float rho = sqrt(x * x + y * y + z * z);
float normTheta =
getTheta(x, y) / (2 * M_PI); 
float normPhi = (M_PI - acos(z / rho)) / M_PI; 

float iX = normTheta * inputWidth;
float iY = normPhi * inputHeight;

if (iX >= inputWidth) {
iX = iX - (inputWidth);
}
if (iY >= inputHeight) {
iY = iY - (inputHeight);
}

destination.at<cv::Vec3b>(i, j) = source.at<cv::Vec3b>(int(iY), int(iX));
}
}
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> diff = end - begin;

cv::imshow("Processed Image", destination);
std::string output_file(argv[2]);
cv::imwrite(output_file, destination);

cout << "Processing time: " << diff.count() << " s" << endl;

cv::waitKey();
return 0;
}
