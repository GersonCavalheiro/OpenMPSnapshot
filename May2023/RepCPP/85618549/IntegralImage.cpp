

#include "IntegralImage.h"
#include <x86intrin.h>
#include <opencv/highgui.h>

void IntegralImage::vIntegralImage(cv::Mat& imgsrc, cv::Mat& integralImg)
{
integralImg.create( imgsrc.rows, imgsrc.cols, CV_32FC1 );

int col, row ;
#pragma omp parallel for private( col, row )
for( col = 0 ; col < imgsrc.cols-3 ; col=col+4 ) {
__m128 sumCurrSSE = _mm_set1_ps( 0 ) ;
for( row = 0 ; row < imgsrc.rows ; ++row ) {
sumCurrSSE = _mm_add_ps( sumCurrSSE,
_mm_setr_ps( imgsrc.at<float>( row, col ),
imgsrc.at<float>( row, col+1 ),
imgsrc.at<float>( row, col+2 ),
imgsrc.at<float>( row, col+3 ) ) ) ;

_mm_storeu_ps( (float*)&integralImg.at<float>( row, col ), sumCurrSSE ) ;
}
}

for( col = (imgsrc.cols/4)*4 ; col < imgsrc.cols ; ++col ) {
float sumCurr = 0 ;
for( row = 0 ; row < imgsrc.rows ; ++row ) {
sumCurr += imgsrc.at<float>( row, col ) ;
integralImg.at<float>( row, col ) = sumCurr ;
}
}
}

void IntegralImage::uIntegralImageR(cv::Mat& imgsrc, cv::Mat& integralImg)
{
integralImg.create( imgsrc.rows, imgsrc.cols, CV_32FC1 );

int col, row ;
#pragma omp parallel for private( col, row )
for( row = 0 ; row < imgsrc.rows-3 ; row=row+4 ) {
__m128 sumCurrSSE = _mm_set1_ps( 0 ) ;
for( col = imgsrc.cols-1 ; col >= 0 ; --col ) {
sumCurrSSE = _mm_add_ps( sumCurrSSE,
_mm_setr_ps( imgsrc.at<float>( row, col ),
imgsrc.at<float>( row+1, col ),
imgsrc.at<float>( row+2, col ),
imgsrc.at<float>( row+3, col ) ) );

float tmparr[4];
_mm_storeu_ps( (float*)&tmparr, sumCurrSSE ) ;

integralImg.at<float>( row, col ) = tmparr[0];
integralImg.at<float>( row+1, col ) = tmparr[1];
integralImg.at<float>( row+2, col ) = tmparr[2];
integralImg.at<float>( row+3, col ) = tmparr[3];
}
}

for( row = (imgsrc.rows/4)*4 ; row < imgsrc.rows ; ++row ) {
float sumCurr = 0 ;
for( col = imgsrc.cols-1 ; col >= 0 ; --col ) {
sumCurr += imgsrc.at<float>( row, col ) ;
integralImg.at<float>( row, col ) = sumCurr ;
}
}
}

void IntegralImage::vWeight(cv::Mat& imgsrc, cv::Mat& wImg)
{
wImg.create( imgsrc.rows, imgsrc.cols, CV_32FC1 ) ;

int col, row ;
#pragma omp parallel for private( col, row )
for( col = 0 ; col < imgsrc.cols-3 ; col=col+4 )
{
__m128 sumTotSSE = _mm_setr_ps( imgsrc.at<float>( imgsrc.rows-1, col ),
imgsrc.at<float>( imgsrc.rows-1, col+1 ),
imgsrc.at<float>( imgsrc.rows-1, col+2 ),
imgsrc.at<float>( imgsrc.rows-1, col+3 ) );

__m128 dbz_mask = _mm_cmpneq_ps(sumTotSSE,  _mm_set1_ps(0) );

for( row = 0 ; row < imgsrc.rows ; ++row )
{
__m128 sumCurrSSE = _mm_setr_ps(imgsrc.at<float>( row, col ),
imgsrc.at<float>( row, col+1 ),
imgsrc.at<float>( row, col+2 ),
imgsrc.at<float>( row, col+3 ) );

__m128 vWeightSSE = _mm_div_ps(sumCurrSSE, sumTotSSE);

vWeightSSE = _mm_and_ps(vWeightSSE, dbz_mask);

_mm_storeu_ps((float*)&wImg.at<float>( row, col ), vWeightSSE);
}
}

for( col = (imgsrc.cols/4)*4 ; col < imgsrc.cols ; ++col )
{
float sumTot =  imgsrc.at<float>( imgsrc.rows-1, col );

for( row = 0 ; row < imgsrc.rows ; ++row )
{
if(sumTot < 0.00001){
wImg.at<float>( row, col ) = 0;
}
else{
float sumCurr = imgsrc.at<float>( row, col ) ;
wImg.at<float>( row, col ) = sumCurr/sumTot ;
}
}
}
}


void IntegralImage::uWeightR(cv::Mat& imgsrc, cv::Mat& wImg)
{
wImg.create( imgsrc.rows, imgsrc.cols, CV_32FC1 ) ;

int col, row ;
#pragma omp parallel for private( col, row )
for( row = 0 ; row < imgsrc.rows-3 ; row=row+4 )
{
__m128 sumTotSSE = _mm_setr_ps( imgsrc.at<float>( row, 0 ),
imgsrc.at<float>( row+1, 0 ),
imgsrc.at<float>( row+2, 0 ),
imgsrc.at<float>( row+3, 0 ) );

__m128 dbz_mask = _mm_cmpneq_ps(sumTotSSE,  _mm_set1_ps(0) );

for( col = imgsrc.cols-1 ; col >= 0 ; --col )
{
__m128 sumCurrSSE = _mm_setr_ps(imgsrc.at<float>( row, col ),
imgsrc.at<float>( row+1, col ),
imgsrc.at<float>( row+2, col ),
imgsrc.at<float>( row+3, col ) );

__m128 uWeightSSE = _mm_div_ps(sumCurrSSE, sumTotSSE);

uWeightSSE = _mm_and_ps(uWeightSSE, dbz_mask);

float tmparr[4];
_mm_storeu_ps((float*)&tmparr, uWeightSSE);
wImg.at<float>( row, col ) = tmparr[0];
wImg.at<float>( row+1, col ) = tmparr[1];
wImg.at<float>( row+2, col ) = tmparr[2];
wImg.at<float>( row+3, col ) = tmparr[3];
}
}

for( row = (imgsrc.cols/4)*4 ; row < imgsrc.rows ; ++row )
{
float sumTot =  imgsrc.at<float>( row, 0 );

for( col = imgsrc.cols-1 ; col >= 0 ; --col )
{
if(sumTot < 0.00001){
wImg.at<float>( row, col ) = 0;
}
else{
float sumCurr = imgsrc.at<float>( row, col ) ;
wImg.at<float>( row, col ) = sumCurr/sumTot ;
}
}
}
}

void IntegralImage::mapWeightPC(cv::Mat& imgsrc, cv::Mat& dest, float percentage)
{
dest.create(imgsrc.rows, imgsrc.cols, CV_32FC1);

#pragma omp parallel for
for(int row=0; row < imgsrc.rows; ++row)
{
for(int col=0; col < imgsrc.cols; ++col)
{
float currWeight = imgsrc.at<float>(row, col);
if(currWeight <= percentage){
dest.at<float>(row, col) = currWeight/percentage;
}
else{
dest.at<float>(row, col) = (1-currWeight)/(1-percentage);
}
}
}
}


void IntegralImage::applyWeight(cv::Mat& imgsrc, cv::Mat& weight, cv::Mat& wImg, int times)
{
wImg.create( imgsrc.rows, imgsrc.cols, CV_32FC1 ) ;

cv::Mat tmp;
tmp = imgsrc.clone();

for(int i=0; i<times; ++i){
wImg = tmp.mul(weight);
tmp = wImg;
}
}
