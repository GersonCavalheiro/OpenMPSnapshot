

#include <iostream>
#include "ObservationEquation.h"
#include "CubicBSpline.h"

#define ZERO_CMP 0.00001

int ObservationEquation::countValidPixels(cv::Mat& imgsrc)
{
int nop=0;
#pragma omp parallel for shared(nop)
for(int row=0; row<imgsrc.rows; ++row)
{
for(int col=0; col<imgsrc.cols; ++col)
{
if(imgsrc.at<float>(row, col) > ZERO_CMP){
#pragma omp atomic
++nop;
}
}
}

return nop;
}

ObservationEquation::ObservationEquation(cv::Mat& imgsrc, int numberOfSplines)
{
int numberOfDBP = numberOfSplines+3;
int maxModelDisp = imgsrc.cols;
double knotDistance = (double)maxModelDisp/(double)numberOfSplines;

int numberOfMeasurements = countValidPixels(imgsrc);

H = cv::Mat::zeros(numberOfMeasurements, numberOfDBP, CV_64FC1);
z.create(numberOfMeasurements, 1, CV_64FC1);

int counter=0;

bool isSingular = true;

#pragma omp parallel for
for(int row=0; row<imgsrc.rows; ++row)
{
for(int col=0; col<imgsrc.cols; ++col)
{
if(imgsrc.at<float>(row, col) <= ZERO_CMP){
continue;
}

double disp = col;
int splinePart =  disp/knotDistance; 

if(splinePart == 0){
isSingular=false;
}

double t = disp/knotDistance-splinePart;

cv::Mat baseVals;
CubicBSpline::CUBaseFunctions(t, baseVals);

#pragma omp critical
{
H.at<double>(counter, splinePart) = baseVals.at<double>(0,0)*imgsrc.at<float>(row, col);
H.at<double>(counter, splinePart+1) = baseVals.at<double>(0,1)*imgsrc.at<float>(row, col);
H.at<double>(counter, splinePart+2) = baseVals.at<double>(0,2)*imgsrc.at<float>(row, col);
H.at<double>(counter, splinePart+3) = baseVals.at<double>(0,3)*imgsrc.at<float>(row, col);


z.at<double>(counter, 0) = row*imgsrc.at<float>(row, col);

++counter;
}
}
}

if(isSingular){
std::cout << "Singularity in observation equation!" << std::endl;
}

if(counter != numberOfMeasurements){
std::cout << "Error in Observationequation: Measurement vector has not expected size!" << std::endl;
}
}
