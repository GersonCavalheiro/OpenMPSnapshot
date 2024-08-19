#pragma once

#include "pass_bits/helper/astro_problems/constants.hpp"
#include "pass_bits/problem.hpp"

namespace pass
{

class gtoc1 : public problem
{
public:

std::array<const celestial_body *, 7> sequence;


std::array<bool, 8> rev_flag;


asteroid destination;


double Isp;


double mass;


double DVlaunch;


gtoc1();

double evaluate(const arma::vec &agent) const override;
};
} 
