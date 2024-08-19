#include <iostream>
#include <string>
#include <unordered_set>
#include <map>
#include "k_nearest_neighbors.h"
#include "utils.h"
#include <omp.h>
#include <ctime>

using namespace std;
using namespace Eigen;

KNearestNeighbors::KNearestNeighbors(int k) : k(k) {}

KNearestNeighbors::~KNearestNeighbors() {}

void KNearestNeighbors::fit(MatrixXd X_train, VectorXd y_train){
this->X = X_train;
VectorXi y_ = y_train.cast<int>();
this->y = y_;
this->d = this->X.cols();
unordered_set<int> n_set;
for(int i = 0; i < y_.size(); i++){
n_set.insert(y_(i));
}
this->n = n_set.size();
}



VectorXi KNearestNeighbors::predict(const MatrixXd& X_test, double (*metric)(double* x1, double* x2, int size)){
if(X_test.cols() != d){
throw "dimensionality size not match";
}
VectorXi ret(X_test.rows());
#pragma omp parallel for
for(int i = 0; i < X_test.rows(); i++){
multimap<double, int> dismap; 
for(int j = 0; j < X.rows(); j++){
double distance = metric(Utils::VectorXd_to_double_array(X_test.row(i)), Utils::VectorXd_to_double_array(X.row(j)), d);
dismap.insert(pair<double,int>(distance, j));
}
multimap<double, int>::iterator iter = dismap.begin();
map<int, int> votes; 
for(int m = 0; m < k; m++,iter++){
votes[y[iter->second]]++;
}
int major_label = 0;
for(map<int,int>::iterator iter=votes.begin(); iter != votes.end(); iter++){
if(iter->second > votes[major_label]){
major_label = iter->first;
}
}
ret(i) = major_label;
}
return ret;
}


int KNearestNeighbors::predict(const VectorXd& X_test, double (*metric)(double* x1, double* x2, int size)){
if(X_test.size() != d){
throw "dimensionality size not match";
}
multimap<double, int> dismap; 
for(int j = 0; j < X.rows(); j++){
double distance = metric(Utils::VectorXd_to_double_array(X_test), Utils::VectorXd_to_double_array(X.row(j)), d);
dismap.insert(pair<double,int>(distance, j));
}
multimap<double, int>::iterator iter = dismap.begin();
map<int, int> votes; 

for(int i = 0; i < k; i++, iter++){
votes[y[iter->second]]++;
}
int major_label = 0;
for(map<int,int>::iterator iter=votes.begin(); iter != votes.end(); iter++){
if(iter->second > votes[major_label]){
major_label = iter->first;
}
}
return major_label;
}

MatrixXd KNearestNeighbors::predict_prob(const MatrixXd& X_test, double (*metric)(double* x1, double* x2, int size)){
if(X_test.cols() != d){
throw "dimensionality size not match";
}
MatrixXd ret = MatrixXd::Zero(X_test.rows(), n);
#pragma omp parallel for
for(int i = 0; i < X_test.rows(); i++){
multimap<double, int> dismap; 
for(int j = 0; j < X.rows(); j++){
double distance = metric(Utils::VectorXd_to_double_array(X_test.row(i)), Utils::VectorXd_to_double_array(X.row(j)), d);
dismap.insert(pair<double,int>(distance, j));
}
multimap<double, int>::iterator iter = dismap.begin();
map<int, int> votes; 
for(int m = 0; m < k; m++, iter++){
votes[y[iter->second]]++;
}
multimap<int, double> labelMap;
for(map<int,int>::iterator iter=votes.begin(); iter!=votes.end();iter++){
labelMap.insert(std::pair<int,double>(iter->first,iter->second/double(k)));
}
for(multimap<int,double>::iterator iter=labelMap.begin();iter!=labelMap.end();iter++){
ret(i,iter->first) = iter->second;
}
}
return ret;	
}

