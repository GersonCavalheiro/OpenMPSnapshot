

#include "ElevationMap.h"

#define ZERO_CMP 0.00001

ElevationMap::ElevationMap(float baseWidth_, float focalLength_, float toleranceFactor_)
{
baseWidth = baseWidth_;
focalLength = focalLength_;
toleranceFactor = toleranceFactor_;
}

void ElevationMap::draw(cv::Mat& dispImgOrg, cv::Mat& dispImgUDFiltered, cv::Mat& realImg, RoadRepresentation roadrep, bool segmentRoad)
{
cv::Mat_<cv::Vec3b> realImg_ = realImg;

plainElevationMapColored_ = cv::Mat::zeros(realImg_.size(), realImg_.type());
plainElevationMap= cv::Mat::zeros(realImg_.size(), CV_32FC1);

#pragma omp parallel for
for(int row=0; row<roadrep.validSampleRange.maxValue; ++row)
{
for(int col=0; col<dispImgOrg.cols; ++col)
{
float currDisp = dispImgOrg.at<float>(row, col) + 0.5;

if((int)currDisp==0){
continue;
}

if(currDisp <= roadrep.validSampleRange.minIndex+1){
continue;
}

bool tooClose=false;
int tooCloseTolerance = 5;

if(currDisp > roadrep.validSampleRange.maxIndex + tooCloseTolerance){
tooClose=true;
}

if(currDisp > roadrep.validSampleRange.maxIndex){
currDisp = roadrep.validSampleRange.maxIndex;
}

int currRowOfDisp = roadrep.LUT_rowOfDisp.at<int>(currDisp, 0);

bool udf = false;
if(currDisp > 0 && (int)dispImgUDFiltered.at<float>(row, col) == 0){
udf = true;
}

if(segmentRoad && !udf && !tooClose && row > currRowOfDisp-toleranceFactor*row && row < currRowOfDisp+toleranceFactor*row)
{
plainElevationMapColored_(row, col)[0] = 100;
plainElevationMap.at<float>(row, col) = 0;
}

else if(segmentRoad && !tooClose && row > currRowOfDisp-toleranceFactor*row)
{
continue;

}

else
{
int dispIndex = currDisp+0.5;
if(dispIndex >= roadrep.validSampleRange.maxIndex){
dispIndex = roadrep.validSampleRange.maxIndex-1;
}
int currRowOfDisp = roadrep.LUT_rowOfDisp.at<int>(dispIndex, 0);

float alpha = acos((row*currRowOfDisp + focalLength*focalLength) / (sqrt(row*row+focalLength*focalLength) * sqrt(currRowOfDisp*currRowOfDisp + focalLength*focalLength)));

float r1 = baseWidth*focalLength/currDisp;
float r2 = r1; 

float height = sqrt(r1*r1 + r2*r2 - 2*r1*r2*cos(alpha));

plainElevationMap.at<float>(row, col) = height;

float maxHeight = 1.5;

float H = 120*(1-height/maxHeight) ;
H = H < 0 ? 0 : H;
float S = 1;
float V = 1;

float R=0, G=0, B=0;

int h = H/60;
float f = (float)H/60. - h;
float p = V*(1-S);
float q = V*(1-S*f);
float t = V*(1-S*(1-f));

if(h==0 || h==6){
R=V;
G=t;
B=p;
}
else if(h==1){
R=q;
G=V;
B=p;
}
else if(h==2){
R=p;
G=V;
B=t;
}
else if(h==3){
R=p;
G=q;
B=V;
}
else if(h==4){
R=t;
G=p;
B=V;
}
else if(h==5){
R=V;
G=p;
B=q;
}

float fac=100;
plainElevationMapColored_(row, col)[0] = B*fac;
plainElevationMapColored_(row, col)[1] = G*fac;
plainElevationMapColored_(row, col)[2] = R*fac;
}
}
}



#pragma omp parallel for
for(int row=0; row < realImg_.rows; ++row)
{
for(int col=0; col < realImg_.cols; ++col)
{
float sum;

sum = realImg_(row, col)[0] + plainElevationMapColored_(row, col)[0];
sum = sum > 255 ? 255 : sum;
realImg_(row, col)[0] = sum;

sum = realImg_(row, col)[1] + plainElevationMapColored_(row, col)[1];
sum = sum > 255 ? 255 : sum;
realImg_(row, col)[1] = sum;

sum = realImg_(row, col)[2] + plainElevationMapColored_(row, col)[2];
sum = sum > 255 ? 255 : sum;
realImg_(row, col)[2] = sum;
}
}
}
