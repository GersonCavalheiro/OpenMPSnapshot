#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class sum_of_different_powers_function : public problem
{
public:

explicit sum_of_different_powers_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
