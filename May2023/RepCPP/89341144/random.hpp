#pragma once

#include <armadillo> 
#include <vector>    

namespace pass
{

double random_double_uniform_in_range(double min, double max);


int random_integer_uniform_in_range(int min, int max);


arma::rowvec integers_uniform_in_range(const int min, const int max, const int count);


arma::vec random_neighbour(const arma::vec &agent,
const double minimal_distance,
const double maximal_distance);

} 
