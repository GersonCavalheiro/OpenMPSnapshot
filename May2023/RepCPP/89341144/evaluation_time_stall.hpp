#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class evaluation_time_stall : public problem
{
public:

const pass::problem &wrapped_problem;


arma::uword repetitions;


explicit evaluation_time_stall(const pass::problem &wrapped_problem);

double evaluate(const arma::vec &agent) const override;
};
} 
