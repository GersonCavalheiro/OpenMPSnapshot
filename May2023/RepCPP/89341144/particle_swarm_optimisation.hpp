#pragma once

#include "pass_bits/optimiser.hpp"

namespace pass
{

class particle_swarm_optimisation : public optimiser
{
public:

arma::uword swarm_size;


double inertia;


double cognitive_acceleration;


double social_acceleration;


double neighbourhood_probability;


particle_swarm_optimisation() noexcept;

virtual optimise_result optimise(const pass::problem &problem);
};
} 
