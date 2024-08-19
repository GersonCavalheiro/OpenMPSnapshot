#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{



class de_jong_function : public problem
{
public:

explicit de_jong_function(const arma::uword dimension);

double evaluate(const arma::vec &agent) const override;
};
} 
