#pragma once

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>

#include <cassert>

template<typename SubElement>
bool equal(cv::Mat_<SubElement> const& lhs, cv::Mat_<SubElement> const& rhs) {
return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template<typename SubElement>
bool assert_mat_equal(cv::Mat_<SubElement> const& lhs, cv::Mat_<SubElement> const& rhs) {
if(!equal(lhs, rhs)) {
std::cout << "LHS: " << lhs << "\nRHS: " << rhs << std::endl;
return false;
}
return true;
}
