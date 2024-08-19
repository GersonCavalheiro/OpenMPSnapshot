#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <iostream>

#include "tracy/Tracy.hpp"
#include "vibe-background-sequential.h"
#include "morphology.h"

void insertionSort(uchar arr[], int n)
{
ZoneScoped
uchar key;

for (int i = 1; i < n; i++)
{
key = arr[i];
int j = i - 1;


while (j >= 0 && arr[j] > key)
{
arr[j + 1] = arr[j];
j = j - 1;
}
arr[j + 1] = key;
}
}

void medianFilter(const cv::Mat& in, cv::Mat& out)
{
assert(in.cols == out.cols);
assert(in.rows == out.rows);

uchar window[9];
#pragma omp parallel for collapse(2) private(window)
for (int i = 1; i < in.rows - 1; i++)
{
for (int j = 1; j < in.cols - 1; j++)
{
window[0] = in.data[(i - 1) * in.step + (j - 1)];
window[1] = in.data[(i - 1) * in.step + j];
window[2] = in.data[(i - 1) * in.step + (j + 1)];
window[3] = in.data[i * in.step + (j + 1)];
window[4] = in.data[i * in.step + j];
window[5] = in.data[i * in.step + (j + 1)];
window[6] = in.data[(i + 1) * in.step + (j - 1)];
window[7] = in.data[(i + 1) * in.step + j];
window[8] = in.data[(i + 1) * in.step + (j + 1)];

insertionSort(window, 9);
out.data[i * out.step + j] = window[4];
}
}
}

void convertRGBToGrayscale(const cv::Mat& in, cv::Mat& out)
{
assert(in.cols == out.cols);
assert(in.rows == out.rows);

#pragma omp parallel for collapse(2)
for (int i = 0; i < in.rows; i++)
{
for (int j = 0; j < in.cols; j++)
{
out.data[i * out.step + j] = 0.114 * in.data[i * in.step + j * 3] + 0.587 * in.data[i * in.step + j * 3 + 1] + 0.299 * in.data[i * in.step + j * 3 + 2];
}
}
}

int drawValidConnectedComponents(const cv::Mat& stats, cv::Mat& frame)
{
ZoneScoped
int peoples = 0;
for (int i = 0; i < stats.rows; i++)
{
const int x = stats.at<int>(cv::Point(0, i));
const int y = stats.at<int>(cv::Point(1, i));
const int w = stats.at<int>(cv::Point(2, i));
const int h = stats.at<int>(cv::Point(3, i));

const bool isAcceptableSize = w < frame.rows - 50 && w > 50;
if (isAcceptableSize)
{
cv::Scalar color(255, 0, 0);
cv::Rect rect(x, y, w, h);
cv::rectangle(frame, rect, color);
peoples++;
}
}
return peoples;
}

int main(int argc, char** argv)
{
cv::Mat kernel1(5, 5, CV_8U, 1);
cv::Mat kernel2(40, 40, CV_8U, 1);
cv::Mat labels;
cv::Mat stats;
cv::Mat centroids;
cv::VideoCapture capture;
vibeModel_Sequential_t* model = nullptr;

capture.open(0);
if (!capture.isOpened())
{
std::cout << "No capture" << std::endl;
return 0;
}
const int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
const int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
cv::Mat frame3C = cv::Mat(height, width, CV_8UC3);
cv::Mat frame1C = cv::Mat(height, width, CV_8UC1);
cv::Mat segmentationMap = cv::Mat(height, width, CV_8UC1);

bool isFirstFrame = true;
std::cout << "Capture is opened" << std::endl;
while (true)
{
FrameMark

capture.read(frame3C);
if (frame3C.empty())
continue;

{
ZoneScopedN("Grayscale")
convertRGBToGrayscale(frame3C, frame1C);
}

if (isFirstFrame)
{
isFirstFrame = false;
model = libvibeModel_Sequential_New();
libvibeModel_Sequential_AllocInit_8u_C1R(model, frame1C.data, frame1C.cols, frame1C.rows);
}

libvibeModel_Sequential_Segmentation_8u_C1R(model, frame1C.data, segmentationMap.data);
libvibeModel_Sequential_Update_8u_C1R(model, frame1C.data, segmentationMap.data);

{
ZoneScopedN("MedianBlur")
medianFilter(segmentationMap, frame1C);
}


{
ZoneScopedN("Opening")
cv::morphologyEx(frame1C, segmentationMap, cv::MORPH_OPEN, kernel1);
}

{
ZoneScopedN("Closing")
cv::morphologyEx(segmentationMap, frame1C, cv::MORPH_CLOSE, kernel2);
}

{
ZoneScopedN("Extract connected components")
cv::connectedComponentsWithStats(frame1C, labels, stats, centroids);
}

int peoples = drawValidConnectedComponents(stats, frame3C);
std::cout << "Peoples detected : " << peoples << std::endl;
cv::imshow("frame", frame3C);
cv::imshow("Segmentation", frame1C);

{
ZoneScopedN("Wait")
if (cv::waitKey(15) >= 0)
break;
}
}
capture.release();
libvibeModel_Sequential_Free(model);

std::cout << "video completed succesfully" << std::endl;
return 0;
}
