#pragma once

#include <opencv2/opencv.hpp>


void bdilate(const cv::Mat& in, cv::Mat& out, const cv::Mat& se);


void berode(const cv::Mat& in, cv::Mat& out, const cv::Mat& se);
