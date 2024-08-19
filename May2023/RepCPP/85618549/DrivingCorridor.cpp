

#include "DrivingCorridor.h"

void DrivingCorridor::filter(cv::Mat& imgsrc, cv::Mat& dest, double steeringAngle)
{
dest = cv::Mat::zeros(imgsrc.rows, imgsrc.cols, CV_32FC1);

double maxAngle=45;
double tmp = imgsrc.cols/4*(steeringAngle/maxAngle);
int lb = imgsrc.cols/4 + tmp;
int rb = imgsrc.cols*3/4 + tmp;

#pragma omp parallel for
for(int row=0; row<imgsrc.rows; ++row)
{
for(int col=lb; col<rb; ++col)
{
dest.at<float>(row, col) = imgsrc.at<float>(row, col);
}
}
}
