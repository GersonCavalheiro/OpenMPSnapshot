#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class search_space_constraint : public problem
{
public:

const pass::problem &wrapped_problem;


search_space_constraint(const pass::problem &wrapped_problem,
arma::uword segment, arma::uword total_segments);

double evaluate(const arma::vec &agent) const override;
};
} 
