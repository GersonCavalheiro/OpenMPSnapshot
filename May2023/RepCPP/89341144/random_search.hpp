#pragma once

#include "pass_bits/optimiser.hpp"

namespace pass
{

class random_search : public optimiser
{
public:

random_search() noexcept;

virtual optimise_result optimise(const pass::problem &problem);
};
} 
