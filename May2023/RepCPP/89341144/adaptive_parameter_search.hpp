#pragma once
#include "pass_bits/problem.hpp"
#include "pass_bits/optimiser.hpp"

namespace pass
{

void search_parameters(const pass::problem &problem, const bool benchmark);


arma::mat parameter_evaluate(pass::optimiser &optimiser, const pass::problem &problems);


arma::uword compare_segments(const arma::mat first_segment_runtimes, const arma::mat second_segment_runtimes);

} 
