

#include <iostream>
#include "KalmanFilter.h"
#include "CubicBSpline.h"
#include <sys/time.h>

using namespace std;

KalmanFilter::KalmanFilter(cv::Mat stateVectorInit, double covarianceOfStateVectorInitScalar_, double processNoiseScalar_, double measurementNoiseScalar_, double b_, double f_, int imageRows_, int imageColumns_)
{
processNoiseScalar = processNoiseScalar_;
measurementNoiseScalar = measurementNoiseScalar_;

b = b_;
f = f_;
imagecenter = imageRows_/2; 
imagecolumns = imageColumns_;
imagerows = imageRows_;

constantPrediction=true;

x = stateVectorInit.clone();
P = cv::Mat::eye(x.rows, x.rows, CV_64FC1) * covarianceOfStateVectorInitScalar_;
Q = cv::Mat::ones(x.rows, x.rows, CV_64FC1) * processNoiseScalar;
}

void KalmanFilter::reset(cv::Mat stateVectorInit, double covarianceOfStateVectorInitScalar_)
{
x = stateVectorInit.clone();
P = cv::Mat::eye(x.rows, x.rows, CV_64FC1) * covarianceOfStateVectorInitScalar_;
Q = cv::Mat::ones(x.rows, x.rows, CV_64FC1) * processNoiseScalar;

constantPrediction=true;
}

void KalmanFilter::kalmanStep(cv::Mat& T, cv::Mat& H_, cv::Mat& z_)
{
H=H_.clone();
z=z_.clone();

predict(T);
update(H_, z_);

constantPrediction=false;
}

void KalmanFilter::predict(cv::Mat& T)
{
if(constantPrediction)
{
P = P + Q;
}

else
{
splineResampled.create(spline.rows, spline.cols, CV_64FC1);

int invalidSamples=0;
#pragma omp parallel for shared(invalidSamples)
for(int i=0; i<spline.rows; ++i)
{
float d = spline.at<int>(i, 0);
float v = spline.at<int>(i, 1);

if((int)d == 0){
d=0.00000001;
}

float z = b*f/d;
float x = 0;
float y = z/f*(v-imagecenter);

cv::Mat P_old = (cv::Mat_<float>(4, 1) << x, y, z, 1);
cv::Mat P_new = T*P_old;

double d_new = b*f/P_new.at<float>(2,0);
double v_new = f*P_new.at<float>(1,0)/P_new.at<float>(2,0) + imagecenter;

if((int)d_new < imagecolumns && (int)v_new < imagerows){
splineResampled.at<double>(i,0) = d_new;
splineResampled.at<double>(i,1) = v_new;
}
else{
splineResampled.at<double>(i,0) = -1;
splineResampled.at<double>(i,1) = -1;

#pragma omp atomic
++invalidSamples;
}
}

cv::Mat H_resampled = cv::Mat::zeros(splineResampled.rows-invalidSamples, x.rows, CV_64FC1);
cv::Mat z_resampled(splineResampled.rows-invalidSamples, 1, CV_64FC1);

int numberOfSplines = x.rows-3;
double knotDistance = (double)imagecolumns/(double)numberOfSplines;

int counter=0;
#pragma omp parallel for
for(int i=0; i<splineResampled.rows; ++i)
{
double disp = splineResampled.at<double>(i, 0);

if(disp < -0.5){
continue;
}

int splinePart = disp/knotDistance; 

double t = disp/knotDistance-splinePart;

cv::Mat baseVals;
CubicBSpline::CUBaseFunctions(t, baseVals);

#pragma omp critical
{
H_resampled.at<double>(counter, splinePart) = baseVals.at<double>(0,0);
H_resampled.at<double>(counter, splinePart+1) = baseVals.at<double>(0,1);
H_resampled.at<double>(counter, splinePart+2) = baseVals.at<double>(0,2);
H_resampled.at<double>(counter, splinePart+3) = baseVals.at<double>(0,3);


z_resampled.at<double>(counter, 0) = splineResampled.at<double>(i, 1);

++counter;
}
}

cv::Mat x_predict(8,1,CV_64FC1);

if(!solve(H_resampled, z_resampled, x_predict, cv::DECOMP_SVD)){
std::cout << "Error Solving System!" << std::endl;
}

cv::Mat F = (x_predict.mul(1/x)*cv::Mat::ones(1, x.rows, CV_64FC1)).mul(cv::Mat::eye(x.rows, x.rows, CV_64FC1));

cv::Mat P_predict = F*P*F.t() + Q;


P = P_predict.clone();
x = x_predict.clone();
}
}

void KalmanFilter::update(cv::Mat& H_, cv::Mat& z_)
{
R = cv::Mat::eye(z_.rows, z_.rows, CV_64FC1) * measurementNoiseScalar;
R_inv = cv::Mat::eye(z_.rows, z_.rows, CV_64FC1) * 1./measurementNoiseScalar;

cv::Mat Ht = H.t(); 



cv::Mat P_inv = P.inv();

cv::Mat Y = P_inv;
cv::Mat I = Ht*R_inv*H;
cv::Mat y = P_inv*x;
cv::Mat i = Ht*R_inv*z;

Y = Y + I;
y = y + i;

cv::Mat P_ = Y.inv();
cv::Mat x_ = P_*y;

P = P_.clone();
x = x_.clone();

CubicBSpline::getSample(x, 0.01, imagecolumns, spline);
}

void KalmanFilter::getStateVector(cv::Mat& x_)
{
x_ = x;
}

void KalmanFilter::getSplineSample(cv::Mat& S_)
{
S_ = spline;
}
