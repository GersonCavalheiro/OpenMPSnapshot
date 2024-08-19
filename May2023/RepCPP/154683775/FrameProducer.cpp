#include "FrameProducer.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include <omp.h>

#define CV_CAP_PROP_FOURCC 6
#define CV_CAP_PROP_FRAME_COUNT 7
#define CV_BGR2GRAY 6

using namespace cv;
using namespace std;


FrameProducer::FrameProducer(const string fname, FrameBuffer& fbuf)
: buf(fbuf) {
capture = VideoCapture(fname);
my_frame_count = capture.get(CV_CAP_PROP_FRAME_COUNT);
fourcc = capture.get(CV_CAP_PROP_FOURCC);
}


void FrameProducer::convertFrames(const int frame_count, Size resolution) {
Mat original, grayscale, resized;
for (int i = 0; i < frame_count; i++) {
capture.read(original);


if (original.empty()) {
break;
}


cvtColor(original, grayscale, CV_BGR2GRAY);


resize(grayscale, resized, resolution);


vector<uchar> array;
array.assign(resized.datastart, resized.dataend);
for (int i = 0; i < resized.rows; i++) {
auto g_i = resized.ptr<uchar>(i);

#pragma omp simd
for (int j = 0; j < resized.cols; j++) {
g_i[j] = g_i[j] ^ 0xFF;
}
}

buf.addFrame(resized);
}
capture.release();
}