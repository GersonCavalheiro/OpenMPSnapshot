#include <iostream>
#include "utils.h"
#include "naive_bayes.h"
#include <omp.h>
#include <ctime>
#include <cmath>


using namespace std;
using namespace Eigen;

NaiveBayes::NaiveBayes() {}

NaiveBayes::~NaiveBayes() {}

void NaiveBayes::fit(const MatrixXd& X, const VectorXd& y){
d = X.cols();
int n = y.size();
VectorXi y_ = y.cast<int>();
for(int i = 0; i < n; i++){
++priors[y_(i)];
}
auto tmp = priors;
for(auto p = priors.begin(); p != priors.end(); p++){
cout << p->first << " " << p->second << endl;
p->second = p->second/n;
}
for(auto iter = tmp.begin(); iter != tmp.end(); iter++){
MatrixXd X_c = MatrixXd::Ones(static_cast<int>(iter->second), X.cols());
cout << X_c.rows() << " " << X_c.cols() << endl;
for(int i = 0, k = 0; i < y_.rows(); i++){
if(y_(i) == iter->first){
X_c.row(k) = X.row(i);
k++;
}
}
VectorXd X_c_mean = X_c.colwise().mean();
VectorXd X_c_var = VectorXd::Ones(X_c_mean.size());
#pragma omp parallel for
for(int j = 0; j < X_c.cols(); j++){
X_c_var(j) = (X_c.col(j).array() - X_c_mean(j)).abs2().mean();
}

map<int, pair<double, double> > parameter; 
for(int j = 0; j < X_c_mean.size(); j++){
parameter.insert(pair<int, pair<double, double> >(j, pair<double, double>(X_c_mean(j), X_c_var(j))));
}
parameters.insert(pair<int, map<int, pair<double, double>>>(iter->first, parameter)); 
}
}

VectorXi NaiveBayes::predict(const MatrixXd& X){
if (X.cols() != d){
throw "dimensionality size not match";
}
VectorXi ret(X.rows());
#pragma omp parallel for
for(int i = 0; i < X.rows(); i++){
ret(i) = predict(VectorXd(X.row(i)));
}
return ret;
}


int NaiveBayes::predict(const VectorXd& X){
map<int, double> posteriors; 
for(auto p = parameters.begin(); p != parameters.end(); p++){
double posterior = priors[p->first];
auto q = p->second;
for(auto qq = q.begin(); qq != q.end(); qq++){
double likelihood = calculate_likelihood(qq->second.first, qq->second.second, X(qq->first));
posterior *= likelihood;
}
posteriors.insert(pair<int, double>(p->first, posterior));
}
double posterior = 0.0;
int label = 0;
for(auto p = posteriors.begin(); p != posteriors.end(); p++){
if(p->second > posterior){
posterior = p->second;
label = p->first;
}
}
return label;
}

double NaiveBayes::calculate_likelihood(double mean, double var, double x){
double eps = 1e-4;
double coeff = 1.0 / sqrt(2.0 * M_PI * var + eps);
double exponent = exp(-(pow((x - mean), 2) / (2 * var + eps)));
return coeff * exponent;
}