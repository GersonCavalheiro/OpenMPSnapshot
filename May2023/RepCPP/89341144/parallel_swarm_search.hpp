#pragma once

#include "pass_bits/optimiser.hpp"

namespace pass
{

class parallel_swarm_search : public optimiser
{
public:

arma::uword swarm_size;


double inertia;


double cognitive_acceleration;


double social_acceleration;


double neighbourhood_probability;

#if defined(SUPPORT_MPI)

arma::uword migration_stall;
#endif


parallel_swarm_search() noexcept;

virtual optimise_result optimise(const pass::problem &problem);

private:

#if defined(SUPPORT_OPENMP)
int number_threads;
#endif
};
} 
