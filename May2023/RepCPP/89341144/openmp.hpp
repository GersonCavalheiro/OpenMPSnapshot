#pragma once
#include "pass_bits/problem.hpp"

namespace pass
{

bool enable_openmp(const pass::problem &problem);


arma::mat train(const int &examples);


arma::rowvec build_model(const arma::mat &training_points);

} 
