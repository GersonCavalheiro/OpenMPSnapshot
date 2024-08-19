#pragma once

#include "pass_bits/problem.hpp"

namespace pass
{

class messenger_full : public problem
{
public:

messenger_full();

double evaluate(const arma::vec &agent) const override;
};
} 
