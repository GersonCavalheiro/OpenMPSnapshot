
#include <iostream>
#include <vector>
#include <omp.h>
#include <numeric>
#include "LR.hpp"

namespace LR {

double g(double x) {
return 1.0 / (1.0 + std::exp(-x));
}

VectorXd LogisticRegression::predict(MatrixXd test) {
return (test * theta).unaryExpr(std::ptr_fun(g));
}

void LogisticRegression::regularize(VectorXd &gw) {
gw += this->lambda / this->m * this->theta;
gw[0] -= this->lambda / this->m * this->theta(0);
}

void LogisticRegression::fit_vec(std::pair<Eigen::MatrixXd, Eigen::VectorXd> train) {
m = train.first.rows();
n = train.first.cols();

theta.setZero(n);

for (size_t i = 0; i < maxIter; i++) {
VectorXd inner = train.first * theta;

VectorXd ga = inner.unaryExpr(std::ptr_fun(g));

VectorXd gw = 1.0 / this->m * (train.first.transpose() * (ga - train.second));

regularize(gw);

this->theta -= this->alpha * gw;

if (verbose) {
std::cout << "gradient " << gw.norm() << std::endl;
}

if (this->epsilon > gw.norm()) {
break;
}
}
}

void LogisticRegression::fit_naive(std::pair<MatrixXd, VectorXd> train) {
m = train.first.rows();
n = train.first.cols();

theta.setZero(n);

for (size_t iter = 0; iter < maxIter; iter++) {

VectorXd gw;
gw.setZero(n);
for (size_t i = 0; i < m; i++) {
double coeff = train.first.row(i) * theta;

coeff = 1.0 / m * (g(coeff) - train.second(i));

VectorXd gwi = coeff * train.first.row(i);

gw += gwi;
}

regularize(gw);

this->theta -= this->alpha * gw;

if (verbose) {
std::cout << "gradient " << gw.norm() << std::endl;
}

if (this->epsilon > gw.norm()) {
break;
}
}

}

void LogisticRegression::fit_parallel(std::pair<MatrixXd, VectorXd> train) {
m = train.first.rows();
n = train.first.cols();

theta.setZero(n);

int coreNum = omp_get_num_procs();

std::vector<VectorXd> sumInCore(coreNum);

std::vector<size_t> iters(coreNum);

size_t sectionNum = m / coreNum;

for (size_t iter = 0; iter < maxIter; iter++) {

for (auto &item : sumInCore) {
item.setZero(n);
}

VectorXd gw;
gw.setZero(n);

#pragma omp parallel for
for (size_t core = 0; core < coreNum; core++) {

for (iters[core] = core * sectionNum; iters[core] < (core + 1) * sectionNum && iters[core] < m; iters[core]++) {

double coeff = train.first.row(iters[core]) * theta;

coeff = 1.0 / m * (g(coeff) - train.second(iters[core]));

sumInCore[core] += coeff * train.first.row(iters[core]);

}

}

gw = std::accumulate(sumInCore.begin(), sumInCore.end(), gw);

regularize(gw);

this->theta -= this->alpha * gw;

if (verbose) {
std::cout << "gradient " << gw.norm() << std::endl;
}

if (this->epsilon > gw.norm()) {
break;
}
}

}

}
