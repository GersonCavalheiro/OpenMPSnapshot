#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{

class rosetta : public problem
{
public:

rosetta();

double evaluate(const arma::vec &agent) const override;
};
} 
