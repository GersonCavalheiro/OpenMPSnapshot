#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <x86intrin.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <string>
#include <sstream>
#include <sys/time.h>
#include "src/DisparityHistogram.h"
#include "src/IntegralImage.h"
#include "src/MEstimator.h"
#include "src/CubicBSpline.h"
#include "src/KalmanFilter.h"
#include "src/RoadRepresentation.h"
#include "src/ElevationMap.h"
#include "src/EasyVisualOdometry.h"
#include "src/DrivingCorridor.h"
#include "src/DrivingState.h"

using namespace std;

#define CAR

void ownThresholdl2z_i(cv::Mat&, int); 

double t1;
void tic();
double toc();

int main( void )
{
int sequence = 4;
int startimg = 1;
int endimg = 4000;

#ifdef CAR
const int maxModelDisp = 90;			
const int numberOfSplines = 5;
const int imageRows = 391;
const float b = 0.57;					
const float f = 645;
const bool compensateRollAngle=false;
const int udfThresh = 20;				
const int vdThresh = 13; 				
const bool useKF = true;				

const string pathpart = "Sequences/";
#endif
#ifdef MOTORBIKE
const int maxModelDisp = 190;
const int numberOfSplines = 4;
const int imageRows = 720;
const float b = 0.28;
const float f = 798;
const bool compensateRollAngle=true;
const int udfThresh = 30;
const int vdThresh = 5;
const bool useKF = false; 

const string pathpart = "Sequences/motorrad_parts/";
#endif

const int numberOfDBP = numberOfSplines+3;

cv::Mat x(numberOfDBP, 1, CV_64FC1, 1); 
KalmanFilter kf(x, 100, 10000, 0.00001, b, f, imageRows, maxModelDisp);

EasyVisualOdometry viso(f,660,187,b);
DrivingState drivingState;

#ifdef CAR
MEstimator mest(numberOfSplines, 30, 10, 10);
RoadRepresentation roadRepresentation(4,2,5);
ElevationMap elevationMap(b, f, 0.06);
#endif
#ifdef MOTORBIKE
MEstimator mest(numberOfSplines, 30, 10, 4);
RoadRepresentation roadRepresentation(10,2,5);
ElevationMap elevationMap(b, f, 0.04);
#endif

for(int imgindex = startimg; imgindex <= endimg; ++imgindex)
{
char buf[6];
sprintf(buf, "%06d", imgindex);

cout << "Image Index: " << buf << endl;

stringstream path; 
string bufstr(buf);
path << pathpart << sequence << "/";

viso.pushImagePair(path.str() + "I1_" + bufstr + ".png", path.str() + "I2_" + bufstr + ".png");

cv::Mat transMat;

if(imgindex!=startimg)
{
viso.computeStep();
viso.getTransformation(transMat);
}
else{
transMat = (cv::Mat_<float>(4, 4) << 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0);
}


cv::Mat dispImgOrgi, dispImgOrg, tmpImgf, showImg, dispImg_UDFiltered, show_dispImg, realImg;
cv::Mat dispImg4Histograms, dispImg_DCFiltered, dispImg_DC_UDFiltered;
dispImgOrgi = cv::imread(path.str() + bufstr + "_disp.png", 0);
realImg = cv::imread(path.str() + bufstr + "_input.png", 3);

dispImgOrgi.convertTo(dispImgOrg, CV_32FC1);


#ifdef MOTORBIKE
for(int row=540; row<dispImgOrg.rows; ++row)
{
for(int col=0; col<dispImgOrg.cols; ++col)
{
dispImgOrg.at<float>(row, col) = 0;
}
}
#endif

cv::threshold(dispImgOrg, dispImg4Histograms, maxModelDisp-1, -1, cv::THRESH_TOZERO_INV);

if(compensateRollAngle)
{
float rollAngle = DisparityHistogram::estimateRollAngle(dispImg4Histograms, maxModelDisp, -30, 30, 2);
cout << "roll angle: " << rollAngle << endl;

cv::Mat rotImg;
cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(dispImg4Histograms.rows/2., dispImg4Histograms.cols/2.), rollAngle, 1);

cv::warpAffine(dispImg4Histograms, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
dispImg4Histograms = rotImg.clone();

cv::warpAffine(dispImgOrg, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
dispImgOrg = rotImg.clone();

cv::warpAffine(realImg, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
realImg = rotImg.clone();
}

cv::Mat vDispUnfiltered;
DisparityHistogram::calculateVDisparity(dispImg4Histograms, vDispUnfiltered, maxModelDisp);
vDispUnfiltered.convertTo(vDispUnfiltered, CV_32FC1);

#ifdef CAR
drivingState.update(sequence, imgindex);
DrivingCorridor::filter(dispImg4Histograms, dispImg_DCFiltered, drivingState.steeringAngle);
#endif
#ifdef MOTORBIKE
DrivingCorridor::filter(dispImg4Histograms, dispImg_DCFiltered, 0);
#endif

cv::Mat uDisp;
DisparityHistogram::calculateUDisparity(dispImg4Histograms, uDisp, maxModelDisp);

DisparityHistogram::filterObstaclesFromUD(dispImgOrg, uDisp, dispImg_UDFiltered, udfThresh);
DisparityHistogram::filterObstaclesFromUD(dispImg_DCFiltered, uDisp, dispImg_DC_UDFiltered, udfThresh);

cv::Mat vDispOrg_i;
DisparityHistogram::calculateVDisparity(dispImg_DC_UDFiltered, vDispOrg_i, maxModelDisp);

ownThresholdl2z_i(vDispOrg_i, vdThresh);

cv::Mat vIntImg, uIntImg, weight, vDispOrg, vDispWeighted;

vDispOrg_i.convertTo(vDispOrg, CV_32FC1);

IntegralImage::vIntegralImage(vDispOrg, vIntImg);
IntegralImage::vWeight(vIntImg, weight);

showImg = weight;

cv::Mat mappedWeight;
IntegralImage::mapWeightPC(weight, mappedWeight, 0.98);

IntegralImage::applyWeight(vDispOrg, mappedWeight, tmpImgf, 10);

IntegralImage::uIntegralImageR(vDispOrg, uIntImg);
IntegralImage::uWeightR(uIntImg, weight);

IntegralImage::mapWeightPC(weight, mappedWeight, 0.98);

showImg = weight;

IntegralImage::applyWeight(tmpImgf, mappedWeight, vDispWeighted, 10);

cv::Mat c;
mest.estimate(vDispWeighted);

cv::Mat spline;

if(useKF && drivingState.state == DrivingState::straight)
{
kf.kalmanStep(transMat, mest.H_weighted, mest.z_weighted);
kf.getStateVector(c);
kf.getSplineSample(spline);
}
else
{
c = mest.c.clone();
CubicBSpline::getSample(c, 0.01, maxModelDisp, spline);
kf.reset(c, 0.01);
}

cv::cvtColor(vDispUnfiltered, show_dispImg, CV_GRAY2RGB);

roadRepresentation.calculateLUTs(vDispWeighted, c);

float minDisp = roadRepresentation.validSampleRange.minIndex;
cv::line(show_dispImg, cv::Point(minDisp, 0), cv::Point(minDisp, show_dispImg.rows), CV_RGB(0,255,0));


for(int row=0; row<imageRows; ++row)
{
cv::circle(show_dispImg, cv::Point(roadRepresentation.LUT_dispOfRow.at<int>(row, 0), row), 1, CV_RGB(0,0,255));
}


cv::imshow( "Result", show_dispImg/255 ) ;

elevationMap.draw(dispImgOrg, dispImg_UDFiltered, realImg, roadRepresentation, 1);

cv::imshow( "ElevationMap", realImg ) ;
cv::waitKey(1);

if(imgindex==startimg){
cv::waitKey(0);
}
}
}


void ownThresholdl2z_i(cv::Mat& imgsrc, int threshold)
{
#pragma omp parallel for
for( int col = 0 ; col < imgsrc.cols ; ++col )
{
for( int row = 0 ; row < imgsrc.rows ; ++row )
{
if(imgsrc.at<int>(row, col) < threshold){
imgsrc.at<int>(row, col) = 0;
}
}
}
}

void tic()
{
timeval tim;
gettimeofday(&tim, NULL);
t1=tim.tv_sec+(tim.tv_usec/1000000.0);
}

double toc()
{
timeval tim;
double t2;
gettimeofday(&tim, NULL);
t2=tim.tv_sec+(tim.tv_usec/1000000.0);

cout << "time elapsed: " << t2-t1 << endl;

return t2-t1;
}
