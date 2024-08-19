#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class schwefel_function : public problem
{
public:

explicit schwefel_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
