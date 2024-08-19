#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{



class rosenbrock_function : public problem
{
public:

explicit rosenbrock_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
