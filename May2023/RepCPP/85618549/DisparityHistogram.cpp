

#include "DisparityHistogram.h"
#include <x86intrin.h>
#include <iostream>
#include <opencv/highgui.h>
#include <opencv/cv.h>


void DisparityHistogram::calculateVDisparity(cv::Mat& imgsrc, cv::Mat& vDispImg, int maxDisp)
{
vDispImg = cv::Mat::zeros(imgsrc.rows, maxDisp+1, CV_32SC1);

int row, col;
#pragma omp parallel for private( col, row )
for( row = 0 ; row < imgsrc.rows-3 ; row=row+4 )
{
for(col=0; col < imgsrc.cols; ++col)
{
int colindices[4] = {imgsrc.at<float>(row, col) +0.5,
imgsrc.at<float>(row+1, col) +0.5,
imgsrc.at<float>(row+2, col) +0.5,
imgsrc.at<float>(row+3, col) +0.5};

__m128i oldValues = _mm_setr_epi32(	vDispImg.at<int>(row, colindices[0]),
vDispImg.at<int>(row+1, colindices[1]),
vDispImg.at<int>(row+2, colindices[2]),
vDispImg.at<int>(row+3, colindices[3]));

__m128i newValueSSE = _mm_add_epi32(oldValues, _mm_set1_epi32(1));

int tmparr[4];
_mm_storeu_si128( (__m128i*)&tmparr, newValueSSE ) ;

for(int i=0; i<4; ++i)
{
if(colindices[i] != 0 && colindices[i] < maxDisp){
vDispImg.at<int>(row+i, colindices[i]) = tmparr[i];
}
}

}

}

for(row=(imgsrc.rows/4)*4 ; row < imgsrc.rows; ++row)
{
for(col=0; col<imgsrc.cols ; ++col)
{
if((int)imgsrc.at<float>(row, col) == 0){
continue;
}

int colindex = imgsrc.at<float>(row, col) +0.5;
vDispImg.at<int>(row, colindex) = vDispImg.at<int>(row, colindex) + 1;
}
}
}


void DisparityHistogram::calculateUDisparity(cv::Mat& imgsrc, cv::Mat& uDispImg, int maxDisp)
{
uDispImg = cv::Mat::zeros(maxDisp+1, imgsrc.cols, CV_32SC1);

int row, col;
#pragma omp parallel for private( col, row )
for( col = 0 ; col < imgsrc.cols-3 ; col=col+4 )
{
for(row=0; row < imgsrc.rows; ++row)
{
int rowindices[4] = {imgsrc.at<float>(row, col) +0.5,
imgsrc.at<float>(row, col+1) +0.5,
imgsrc.at<float>(row, col+2) +0.5,
imgsrc.at<float>(row, col+3) +0.5};

__m128i oldValues = _mm_setr_epi32(	uDispImg.at<int>(rowindices[0], col),
uDispImg.at<int>(rowindices[1], col+1),
uDispImg.at<int>(rowindices[2], col+2),
uDispImg.at<int>(rowindices[3], col+3));

__m128i newValueSSE = _mm_add_epi32(oldValues, _mm_set1_epi32(1));

int tmparr[4];
_mm_storeu_si128( (__m128i*)&tmparr, newValueSSE ) ;

for(int i=0; i<4; ++i)
{
if(rowindices[i] != 0){
uDispImg.at<int>(rowindices[i], col+i) = tmparr[i];
}
}

}
}

for(col=(imgsrc.cols/4)*4 ; col < imgsrc.cols; ++col)
{
for(row=0; row < imgsrc.rows; ++row)
{
if((int)imgsrc.at<float>(row, col) == 0){
continue;
}

int rowindex = imgsrc.at<float>(row, col) +0.5; 
uDispImg.at<int>(rowindex, col) = uDispImg.at<int>(rowindex, col) + 1;
}
}
}

void DisparityHistogram::filterObstaclesFromUD(cv::Mat& imgsrc, cv::Mat& uDisp, cv::Mat& fImg, int threshold)
{
fImg = imgsrc.clone();

#pragma omp parallel for
for(int col = 0; col < imgsrc.cols ; ++col )
{
for(int row=0; row < imgsrc.rows; ++row)
{
float disp = imgsrc.at<float>(row, col);

if(disp+0.5 > uDisp.rows){
continue;
}

int hVal = uDisp.at<int>(disp+0.5, col);

if(hVal > threshold){
fImg.at<float>(row, col) = 0;
}
}
}
}


float DisparityHistogram::estimateRollAngle(cv::Mat& imgsrc, int maxDisp, float min, float max, float step)
{
cv::Mat rotImg;
cv::Mat vDisp;

double bestscore=0;
float rollAngle=0;

int tmpcounter=1;

for(float angle=min; angle<=max; angle+=step)
{
cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(imgsrc.rows/2., imgsrc.cols/2.), angle, 1);
cv::warpAffine(imgsrc, rotImg, rotMat, imgsrc.size(), cv::INTER_NEAREST);

cv::Mat cutImg = rotImg(cv::Range::all(), cv::Range(260, 660));

calculateVDisparity(cutImg, vDisp, maxDisp);

double score=0;
int scorecounter=0;

for(int row=vDisp.rows/2; row<vDisp.rows; ++row)
{
double maxval=0;
double sum=0;
for(int col=vDisp.cols/2; col<vDisp.cols; ++col)
{
double tmpval = vDisp.at<int>(row, col);
sum += tmpval;

if(tmpval > maxval){
maxval = tmpval;
}
}

if(sum > 0){ 
score = score + maxval*maxval/sum;
++scorecounter;
}
}

score = score/(double)scorecounter;

if(score > bestscore){
bestscore = score;
rollAngle = angle;
}
}

return rollAngle;
}
