#pragma once

#include "cl_mat.hpp"

#include <opencv2/opencv.hpp>

cv::Mat_<cv::Vec4b> sobel(cv::Mat_<cv::Vec4b> RGB);
CLMat<cv::Vec4b> sobel_cl(cv::Mat_<cv::Vec4b> const& input);
