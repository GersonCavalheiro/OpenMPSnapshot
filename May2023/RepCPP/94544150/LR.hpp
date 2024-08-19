
#pragma once

#include <Eigen/Core>

namespace LR {

using Eigen::VectorXd;
using Eigen::MatrixXd;

class LogisticRegression {
private:
double alpha;
double lambda;
double epsilon;
int maxIter;

VectorXd theta;

long m;
long n;

bool verbose;

void regularize(VectorXd &);
public:
LogisticRegression(double alpha, int maxIter, double lambda, bool verbose=false, double epsilon=1e-5)
: alpha(alpha), maxIter(maxIter), lambda(lambda), epsilon(epsilon), verbose(verbose) {}

void fit_vec(std::pair<MatrixXd, VectorXd>);

void fit_naive(std::pair<MatrixXd, VectorXd>);

void fit_parallel(std::pair<MatrixXd, VectorXd>);

VectorXd predict(MatrixXd);
};

}
