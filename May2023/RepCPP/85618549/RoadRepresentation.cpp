

#include <iostream>
#include "RoadRepresentation.h"
#include "CubicBSpline.h"

RoadRepresentation::RoadRepresentation(int a, int b, int c) : validSampleRange(a,b,c)
{
}

void RoadRepresentation::calculateLUTs(cv::Mat& img, cv::Mat& c)
{
cv::Mat sample;
CubicBSpline::getSample(c, 0.01, img.cols, sample);

validSampleRange.findSampleRange(img, sample);

int minDisp = validSampleRange.minIndex;
int maxDisp = validSampleRange.maxIndex;
int maxRow = validSampleRange.maxValue;

int maxModelDisp = img.cols-1;
int imageRows = img.rows;

LUT_rowOfDisp.create(maxModelDisp, 1, CV_32SC1);
LUT_dispOfRow.create(imageRows, 1, CV_32SC1);

#pragma omp parallel for
for(int d=0; d<maxModelDisp; ++d)
{
if(d < minDisp || d > maxDisp){
LUT_rowOfDisp.at<int>(d, 0) = -1;
continue;
}

int currRow = CubicBSpline::cubicBSpline(c, d, maxModelDisp);
if(currRow >= imageRows){
currRow = imageRows-1;
}

LUT_rowOfDisp.at<int>(d, 0) = currRow;
}


#pragma omp parallel for
for(int i=0; i<imageRows; ++i)
{
LUT_dispOfRow.at<int>(i, 0) = -1;
}

#pragma omp parallel for
for(int i=0; i<sample.rows; ++i)
{
int disp = sample.at<int>(i,0);
int row = sample.at<int>(i,1);

if(row < 0 || row >= maxRow || disp < minDisp || disp >= maxDisp){
continue;
}

#pragma omp critical
{
LUT_dispOfRow.at<int>(row, 0) = disp;
}
}

for(int row=1; row<maxRow; ++row)
{
if(LUT_dispOfRow.at<int>(row, 0) == -1){
LUT_dispOfRow.at<int>(row, 0) = LUT_dispOfRow.at<int>(row-1, 0);
}
}
}
