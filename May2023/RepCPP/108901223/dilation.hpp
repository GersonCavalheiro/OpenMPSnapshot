#pragma once

#include "cl_mat.hpp"

#include <opencv2/opencv.hpp>

cv::Mat_<uint8_t> dilate(cv::Mat_<uint8_t> const& input, size_t radius);
cv::Mat_<uint8_t> dilate_omp(cv::Mat_<uint8_t> const& input, size_t radius);
CLMat<uint8_t> dilate_cl(cv::Mat_<uint8_t> const& input, size_t radius);
cv::Mat_<uint8_t> dilate_cv(cv::Mat_<uint8_t> const& input, size_t radius);
