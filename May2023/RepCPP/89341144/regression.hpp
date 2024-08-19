#pragma once
#include <array>
#include <armadillo>
#include <cassert>
namespace pass
{

class regression
{
public:

arma::rowvec linear_model(const arma::rowvec &x_values, const arma::rowvec &y_values);



arma::rowvec poly_model(const arma::rowvec &x_values, const arma::rowvec &y_values, int degree);


double predict_linear(const double &x, const arma::rowvec &model);


double predict_poly(const double &x, const arma::rowvec &model);
};
} 
