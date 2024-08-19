#include <cmath>
#include "utils.h"
#include <iostream>
#include <unordered_map>

using namespace std;
using namespace Eigen;

double Utils::sigmoid(double x){
return 1.0/(1.0+exp(-x));
}

Eigen::VectorXd Utils::sigmoid(Eigen::VectorXd x){
int n = x.size();
Eigen::VectorXd ret(n);
#pragma omp parallel for
for(int i = 0; i < n; i++){
ret(i) = 1.0/(1.0+exp(-x(i)));
}
return ret;
}

double Utils::squareLoss(Eigen::VectorXd y, Eigen::VectorXd y_pred){
int n = y.size();
double loss = 0.0;
#pragma omp parallel for reduction(+: loss)
for(int i = 0; i < n; i++){
loss += pow((y(i) - y_pred(i)), 2);
}
return 0.5 * loss;
}


double Utils::crossEntropyLoss(Eigen::VectorXd y, Eigen::VectorXd y_pred){
int n = y.size();
double loss = 0.0;
#pragma omp parallel for reduction(+: loss)
for(int i = 0; i < n; i++){
double yi_prob = y_pred(i);
yi_prob = std::min(std::max(yi_prob,0.0001),0.9999);
loss -= (y(i)*log2(yi_prob)+(1-y(i))*log2(1-yi_prob));
}
return loss/n;
}


double Utils::accuracy(Eigen::VectorXd y, Eigen::VectorXd pred){
Eigen::VectorXi y_ = y.cast<int>();
int n = y_.size();
double hit = 0.0;
#pragma omp parallel for reduction(+: hit)
for(int i = 0; i < n; i++){
if(y_(i) == (pred(i)>0.5?1:0)){
hit += 1.0;
}
}
return hit/n;
}

double Utils::accuracy(double* y, double* pred, int size){
double hit = 0.0;
#pragma omp parallel for reduction(+: hit)
for(int i = 0; i < size; i++){
if(y[i] == (pred[i]>0.5?1:0)){
hit += 1.0;
}
}
return hit/size;
}

Eigen::MatrixXd Utils::slice(Eigen::MatrixXd X, int start_idx, int end_idx){
Eigen::MatrixXd ret(end_idx-start_idx+1, X.cols());
#pragma omp parallel for
for(int i = start_idx; i <= end_idx; i++){
ret.row(i-start_idx) = X.row(i);
}
return ret;
}

Eigen::VectorXd Utils::slice(Eigen::VectorXd y, int start_idx, int end_idx){
Eigen::VectorXd ret(end_idx-start_idx+1);
#pragma omp parallel for
for(int i = start_idx; i <= end_idx; i++){
ret(i-start_idx) = y(i); 
}
return ret;
}

int* Utils::VectorXi_to_int_array(Eigen::VectorXi y){
int size = y.size();
int* ret = new int[size];
#pragma omp parallel for
for(int i = 0; i < size; i++){
ret[i] = y(i);
}
return ret;
}

double* Utils::VectorXd_to_double_array(Eigen::VectorXd pred){
int size = pred.size();
double* ret = new double[size];
#pragma omp parallel for
for(int i = 0; i < size; i++){
ret[i] = pred(i);
}
return ret;
}



double Utils::euclidean_distance(Eigen::VectorXd x1, Eigen::VectorXd x2){
if(x1.size() != x2.size()){
throw "dimensionality size not match";
}
return (x1-x2).cwiseAbs2().sum();
}

double Utils::euclidean_distance(double* x1, double* x2, int size){
double dis = 0.0;
#pragma omp parallel for reduction(+: dis)
for(int i = 0; i < size; i++){
dis += pow((x1[i] - x2[i]), 2);
}
return sqrt(dis);
}

array<MatrixXd,2> Utils::divide_on_feature(MatrixXd X, int j, double threshold){
int row = X.rows();
int count1 = 0;
int count2 = 0;
for(int i = 0; i < row; i++){
if(X(i,j) >= threshold){
count1++;
}else{
count2++;
}
}
int m = 0;
int k = 0;
MatrixXd X1(count1, X.cols());
MatrixXd X2(count2, X.cols());

for(int i = 0; i < row; i++){
if(X(i,j) >= threshold){
X1.row(m) = X.row(i);
m++;
}else{
X2.row(k) = X.row(i);
k++;
}
}
return array<MatrixXd,2>{X1, X2}; 
}

double Utils::calculate_entropy(VectorXd y){
unordered_map<int, int> labels;
int size = y.size();
for(int i = 0; i < size; i++){
labels[y[i]]++;
}
double entropy = 0.0;
for(auto p = labels.begin(); p != labels.end(); p++){
int count = p->second;
double prob = count/double(size);
entropy += -prob*log2(prob);
}
return entropy;
}

VectorXd Utils::Linear(MatrixXd X, VectorXd y){
return X*y;
}

MatrixXd Utils::calculate_covariance_matrix(MatrixXd X){
int n_samples = X.rows();
MatrixXd X_scaled = X.rowwise() - X.colwise().mean();
MatrixXd covariance_matrix = (1 / (n_samples - 1)) * X_scaled.transpose() * X_scaled;
return covariance_matrix;
}

MatrixXd Utils::calculate_covariance_matrix(MatrixXd X, MatrixXd Y){
int n_samples = X.rows();
MatrixXd X_scaled = X.rowwise() - X.colwise().mean();
MatrixXd Y_scaled = Y.rowwise() - Y.colwise().mean();
MatrixXd covariance_matrix = (1 / (n_samples - 1)) * X_scaled.transpose() * Y_scaled;
return covariance_matrix;
}