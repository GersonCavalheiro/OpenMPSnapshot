#include <iostream>
#include <omp.h>

#include "matrixRoutines.hpp"

int main(){

#pragma omp declare reduction( + : Eigen::MatrixXd : omp_out = omp_out + omp_in) initializer(omp_priv = Eigen::MatrixXd::Zero(5,5))

Eigen::MatrixXd result = Eigen::MatrixXd::Zero(5,5);

#pragma omp parallel for reduction( + : result)
for(size_t i = 0; i < 200; i++){
result += Eigen::MatrixXd::Identity(5,5);
}

printMatrix(result);

}
