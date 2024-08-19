#pragma once

#include <vector>
#include <opencv2/core.hpp>

float calc_GRIC(float *res, float sigma, int n, int model);

void sampsonF_dsqr(std::vector<float> &F, std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int npts, float *res);


void sampsonH_dsqr(std::vector<float> &H, std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int npts, float *res);




void GRIC(std::vector<cv::Point2f> &pts0, std::vector<cv::Point2f> &pts1, int nmatches,
cv::Mat &Fundamental, cv::Mat &Homography, float sigma, float &gricF, float &gricH);
