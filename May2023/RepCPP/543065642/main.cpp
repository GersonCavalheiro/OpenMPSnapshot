#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#include <iostream>
using namespace cv;

using namespace std;

#if defined(_WIN32) 
#if defined(_MSC_VER) || defined(__ICL)
#define __restrict__ __restrict
#endif
#endif

#define PRAGMA(X) _Pragma(#X)
#if defined(__INTEL_COMPILER)
#define __unroll(X) PRAGMA(unroll(X))
#elif defined(__clang__)
#define __unroll(X) PRAGMA(clang loop unroll_count(X))
#elif defined(__GNUC__) || defined(__GNUG__)
#define __unroll(X) PRAGMA(GCC unroll(X))
#else
#define __unroll(X)
#endif

bool testEnv = false; 

void enhanceScannedImage(cv::Mat& input, cv::Mat& output)
{
auto start_time_p = omp_get_wtime();

cv::Mat integralSum;
cv::integral(input, integralSum);

int rows = input.rows;

int cols = input.cols;

int maskWidth = MAX(rows, cols) / 32;
int offset = maskWidth / 2;
double T = 0.15;


#pragma omp parallel for simd schedule(simd:static) shared (rows,cols,offset,T,integralSum)
for (int i = 0; i < rows; ++i)
{

int x1, y1, x2, y2, count, sum;

int* point_1, * point_2;
uchar* p_input, * p_output;

y1 = i - offset;
y2 = i + offset;

if (y1 < 0) { y1 = 0; }
if (y2 >= rows) { y2 = rows - 1; }

point_1 = integralSum.ptr<int>(y1);
point_2 = integralSum.ptr<int>(y2);
p_input = input.ptr<uchar>(i);
p_output = output.ptr<uchar>(i);
for (int j = 0; j < cols; ++j)
{
x1 = j - offset;
x2 = j + offset;

if (x1 < 0) { x1 = 0; }
if (x2 >= cols) { x2 = cols - 1;}

count = (x2 - x1) * (y2 - y1);

sum = point_2[x2] - point_1[x2] - point_2[x1] + point_1[x1];

if ((int)(p_input[j] * count) < (int)(sum * (1.0 - T)))
p_output[j] = 0;
else
p_output[j] = 255;

}
}
auto run_time_p = omp_get_wtime() - start_time_p;
cout << "\n" << "************" << "run_time_Parallel: " << run_time_p << " S " << "***********" << "\n" << endl;
}

int main(int argc, char* argv[])
{
const char* default_file = "bookpage.jpg";
const char* filename = argc >= 2 ? argv[1] : default_file;

cv::Mat src = cv::imread( samples::findFile(filename), cv::ImreadModes::IMREAD_GRAYSCALE);

cv::imshow("original", src);

cv::Mat gray;

if (src.channels() == 3)
{
cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

cv::imshow("gray", gray);
}
else
{
gray = src; 
}
cv::Mat bw1 = cv::Mat::zeros(gray.size(), CV_8UC1);

enhanceScannedImage(gray, bw1);


if (testEnv)
{
cv::imwrite("enhanced_scanned_image.png", bw1);
}
else
{
cv::imshow("enhanced_scanned_image", bw1);
}
cv::waitKey(0);
return 0;
}