

#include "MEstimator.h"
#include "CubicBSpline.h"
#include <opencv/highgui.h>
#include <iostream>

MEstimator::MEstimator(int numberOfSplines_, int maxIterations_, float cauchyFactor_, float errorBound_)
{
numberOfSplines = numberOfSplines_;
numberOfDBP = numberOfSplines_+3;
maxIterations = maxIterations_;
cauchyFactor = cauchyFactor_;
errorBound = errorBound_;
}

void MEstimator::estimate(cv::Mat& imgsrc)
{
ObservationEquation obseq(imgsrc, numberOfSplines);

c.create(numberOfDBP, 1, CV_64FC1);

H_weighted = cv::Mat::zeros(obseq.H.rows, obseq.H.cols, CV_64FC1);
z_weighted = cv::Mat::zeros(obseq.z.rows, obseq.z.cols, CV_64FC1);

errors.create(obseq.z.rows, 1, CV_64FC1);

if(!solve(obseq.H, obseq.z, c, cv::DECOMP_SVD)){
std::cout << "Error Solving System!" << std::endl;
}

cv::Mat c_old = (cv::Mat_<double>(numberOfDBP, 1) << 10000, 10000, 10000, 10000);

int i=0;
for(; i < maxIterations && cv::norm(c, c_old, cv::NORM_INF) > errorBound; ++i)
{
errors = obseq.z - obseq.H*c;

#pragma omp parallel for
for(int j=0; j<obseq.z.rows; ++j)
{
double tmp = (errors.at<double>(j,0)/cauchyFactor);
double weight = 1./(double)(1 + tmp*tmp);

for(int k=0; k<H_weighted.cols; ++k)
{
H_weighted.at<double>(j, k) = obseq.H.at<double>(j, k)*weight;
}
z_weighted.at<double>(j, 0) = obseq.z.at<double>(j, 0)*weight;
}

c_old = c.clone();

if(!solve(H_weighted, z_weighted, c, cv::DECOMP_SVD)){
std::cout << "Error Solving System!" << std::endl;
}

}

}
