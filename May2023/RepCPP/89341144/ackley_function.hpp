#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class ackley_function : public problem
{
public:

explicit ackley_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
