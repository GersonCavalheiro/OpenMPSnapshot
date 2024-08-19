#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/format.hpp>
#include "perceptron.h"
#include "utils.h"
#include <omp.h>
#include <ctime>

using namespace std;
using namespace Eigen;

Perceptron::Perceptron(const int& n_iterations, const double& learning_rate, const double& tolerance, const int& seed, const int& early_stopping_round) : n_iterations(n_iterations), learning_rate(learning_rate), tolerance(tolerance), seed(seed), early_stopping_round(early_stopping_round) {}

Perceptron::~Perceptron(){}

void Perceptron::fit(const MatrixXd& X, const VectorXd& y, VectorXd (*activation)(Eigen::VectorXd x), double (*loss)(VectorXd y, VectorXd y_pred), double (*metric)(VectorXd y, VectorXd pred)){
int n_features = X.cols();
srand(seed);
double limit = 1/sqrt(n_features+1);
W = limit * VectorXd::Random(X.cols()+1);
MatrixXd X_new(X.rows(), X.cols()+1);
X_new<<X, MatrixXd::Ones(X.rows(), 1);

double best_acc = 0.0;
int become_worse_round = 0;

for(int iter = 0; iter < n_iterations; iter++){
VectorXd outputs = X_new*W;
VectorXd y_pred = predict_prob(X);

VectorXd E = y - y_pred;

VectorXd loss_gradient = (-E.array() * Utils::sigmoid(outputs).array()*(VectorXd::Ones(X.rows()) - Utils::sigmoid(outputs)).array()).matrix();
W = W - learning_rate*X_new.transpose()*loss_gradient;

y_pred = predict_prob(X);


double loss = Utils::squareLoss(y, y_pred);
double acc = metric(y, y_pred);
cout << boost::format("Iteration: %d, squareloss:%.5f, accuracy:%.5f") %iter %loss %acc << endl;
if(loss <= tolerance) break;

if(acc < best_acc){
become_worse_round +=1;
}else{
become_worse_round = 0;
best_acc = acc;
}
if(become_worse_round >= early_stopping_round){
cout << "Early stopping. the best accuracy: " << best_acc << endl;
break;
}
}
}

VectorXd Perceptron::predict_prob(const MatrixXd& X){
MatrixXd X_new(X.rows(), X.cols()+1);
X_new << X, MatrixXd::Ones(X.rows(), 1);
VectorXd y_pred_prob = Utils::sigmoid(X_new*W);
return y_pred_prob;
}

VectorXi Perceptron::predict(const MatrixXd& X){
VectorXd ret_ = predict_prob(X);
int n = ret_.size();
VectorXi ret(n);
#pragma omp parallel for
for(int i = 0; i < n; i++){
ret(i) = ret_(i)>0.5?1:0;
}
return ret;
}
