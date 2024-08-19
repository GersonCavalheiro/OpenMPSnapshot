#pragma once

#include <opencv2/opencv.hpp>

double edge_ratio(cv::Mat_<uint8_t> const& dilated_frame,
cv::Mat_<uint8_t> const& threshed_next_frame);

double edge_ratio_omp(cv::Mat_<uint8_t> const& dilated_frame,
cv::Mat_<uint8_t> const& threshed_next_frame);
