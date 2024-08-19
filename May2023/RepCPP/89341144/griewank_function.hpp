#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{



class griewank_function : public problem
{
public:

explicit griewank_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
