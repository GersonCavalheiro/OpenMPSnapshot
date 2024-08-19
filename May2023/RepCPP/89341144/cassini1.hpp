#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{

class cassini1 : public problem
{
public:

cassini1();

double evaluate(const arma::vec &agent) const override;
};
} 
