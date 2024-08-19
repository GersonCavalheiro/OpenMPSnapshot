#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include "logistic_regression.h"
#include "utils.h"
#include <omp.h>
#include <ctime>

using namespace std;
using namespace Eigen;

LogisticRegression::LogisticRegression() : max_iter(1000), alpha(0.01), lambda(0.0), tolerance(1e-7), seed(2018), use_batch(false) {}

LogisticRegression::LogisticRegression(const int& max_iter, const double& alpha, const double& lambda, const double& tolerance, const int& seed, bool use_batch): max_iter(max_iter), alpha(alpha), lambda(lambda), tolerance(tolerance), seed(seed), use_batch(use_batch){}

LogisticRegression::~LogisticRegression(){}


void LogisticRegression::fit(const MatrixXd& X, const VectorXd& y, const int& batch_size, const int& early_stopping_round, double (*metric)(double* y, double* pred, int size)){
srand(seed);
W = VectorXd::Random(X.cols()+1);   
MatrixXd X_new(X.rows(), X.cols()+1);
X_new<<X, MatrixXd::Ones(X.rows(),1);
double best_acc = -1.0;
double best_loss = -1.0;
MatrixXd bestW = W;
int become_worse_round = 0;

if(use_batch){
MatrixXd X_batch;
VectorXd y_batch;
MatrixXd X_new_batch;
for(int iter = 0; iter < max_iter; iter++){
int start_idx = (batch_size*iter)%(static_cast<int>(X.rows()));
int end_idx = min(start_idx+batch_size, static_cast<int>(X.rows()));

X_batch = Utils::slice(X, start_idx, end_idx-1);
y_batch = Utils::slice(y, start_idx, end_idx-1);
X_new_batch = Utils::slice(X_new, start_idx, end_idx-1);

VectorXd y_pred = predict_prob(X_batch);
VectorXd E = y_pred - y_batch;

W -= (alpha * X_new_batch.transpose() * E + lambda * W);
y_pred = predict_prob(X_batch);

double loss = Utils::crossEntropyLoss(y_batch, y_pred);
double acc = metric(Utils::VectorXd_to_double_array(y_batch), Utils::VectorXd_to_double_array(y_pred), end_idx-start_idx);
cout << boost::format("Iteration: %d, logloss:%.5f, accuracy:%.5f") %iter %loss %acc << endl;

if(loss <= tolerance) break;

if(acc < best_acc){
become_worse_round += 1;
}else{
become_worse_round = 0;
best_acc = acc;
bestW = W;
best_loss = loss;
}
if(become_worse_round >= early_stopping_round){
W = bestW;
cout << boost::format("Early stopping in %d, the best iteration is: %d, logloss: %.5f, accuracy:%.5f") %iter %(iter-early_stopping_round) %best_loss %best_acc << endl;
cout << "Early stopping." << endl;
break;
}
}
}
else{
for(int iter = 0; iter < max_iter; iter++){

VectorXd y_pred = predict_prob(X);
VectorXd E = y_pred - y;

W -= (alpha * X_new.transpose() * E + lambda * W);
y_pred = predict_prob(X);

double loss = Utils::crossEntropyLoss(y, y_pred);
double acc = metric(Utils::VectorXd_to_double_array(y), Utils::VectorXd_to_double_array(y_pred), X.rows());
cout << boost::format("Iteration: %d, logloss:%.5f, accuracy:%.5f") %iter %loss %acc << endl;

if(loss <= tolerance) break;

if(acc < best_acc){
become_worse_round += 1;
}else{
become_worse_round = 0;
best_acc = acc;
bestW = W;
best_loss = loss;
}
if(become_worse_round >= early_stopping_round){
W = bestW;
cout << boost::format("Early stopping in %d, the best iteration is: %d, logloss: %.5f, accuracy:%.5f") %iter %(iter-early_stopping_round) %best_loss %best_acc << endl;
cout << "Early stopping." << endl;
break;
}
}
}
}

VectorXd LogisticRegression::predict_prob(const MatrixXd& X){
MatrixXd X_new(X.rows(), X.cols()+1);
X_new << X, MatrixXd::Ones(X.rows(),1);
int num_samples = X_new.rows();
VectorXd y_pred_prob = VectorXd::Zero(num_samples);
#pragma omp parallel for
for(int num = 0; num < num_samples; num++){
y_pred_prob(num) = Utils::sigmoid(X_new.row(num).dot(W));
}
return y_pred_prob;
}

VectorXi LogisticRegression::predict(const MatrixXd& X){
VectorXd y_pred_prob = predict_prob(X);
VectorXi y_pred(y_pred_prob.size());
#pragma omp parallel for
for(int num = 0; num < y_pred_prob.size(); num++){
y_pred(num) = y_pred_prob(num)>0.5?1:0;
}
return y_pred;
}

VectorXd LogisticRegression::getW(){
return W;
}

void LogisticRegression::saveWeights(const std::string& fpath){
std::ofstream ofile;
ofile.open(fpath.c_str());
if (!ofile.is_open()){
std::cerr << "Can not open the file when call LogisticRegression::saveWeights" << std::endl;
return;
}
for(int i = 0; i < W.size() - 1; i++){
ofile << W(i) << " ";
}
ofile << W(W.size()-1);
ofile.close();
}

void LogisticRegression::loadWeights(const std::string& fpath){
std::ifstream ifile;
ifile.open(fpath.c_str());
if (!ifile.is_open()){
std::cerr << "Can not open the file when call LogisticRegression::loadWeights" << std::endl;
return;
}
std::string line;
std::vector<double> weights;
getline(ifile, line);   
std::stringstream ss(line);
double tmp;
while(!ss.eof()){
ss >> tmp;
weights.push_back(tmp);
}
W = VectorXd::Map(weights.data(), weights.size());
ifile.close();
}
