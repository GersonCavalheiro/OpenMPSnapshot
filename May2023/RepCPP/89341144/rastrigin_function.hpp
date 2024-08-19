#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{


class rastrigin_function : public problem
{
public:

explicit rastrigin_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
