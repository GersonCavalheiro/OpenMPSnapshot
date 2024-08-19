#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

#include "scale.hpp"
#include "GRIC.hpp"
#include "pose.hpp"
#include "tracker.hpp"
#include "detector.hpp"
#include "viewer.hpp"

class ARM_VO
{
public:

ARM_VO(void); 
ARM_VO(const std::string& paramsFileName); 
void loadSetting(const std::string& paramsFileName);
void init(const cv::Mat& firstFrame); 
void update(const cv::Mat& currentFrame); 

bool initialized = false;
cv::Mat R_f, t_f; 
std::vector<cv::Point2f> prev_inliers, curr_inliers; 

private:

int maxFeatures; 
cv::Mat cameraMatrix;
cv::Mat prev_frame;
std::vector<cv::Point2f> prev_keypoints;
gridFASTdetector Detector;
tracker KLT;
scaleEstimator Scale;
};
