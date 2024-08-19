



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cstdint.h>

#include <hydra/detail/external/hydra_thrust/random/discard_block_engine.h>
#include <hydra/detail/external/hydra_thrust/random/linear_congruential_engine.h>
#include <hydra/detail/external/hydra_thrust/random/linear_feedback_shift_engine.h>
#include <hydra/detail/external/hydra_thrust/random/subtract_with_carry_engine.h>
#include <hydra/detail/external/hydra_thrust/random/xor_combine_engine.h>

#include <hydra/detail/external/hydra_thrust/random/uniform_int_distribution.h>
#include <hydra/detail/external/hydra_thrust/random/uniform_real_distribution.h>
#include <hydra/detail/external/hydra_thrust/random/normal_distribution.h>

namespace hydra_thrust
{






namespace random
{




typedef discard_block_engine<ranlux24_base, 223, 23> ranlux24;



typedef discard_block_engine<ranlux48_base, 389, 11> ranlux48;



typedef xor_combine_engine<
linear_feedback_shift_engine<hydra_thrust::detail::uint32_t, 32u, 31u, 13u, 12u>,
0,
xor_combine_engine<
linear_feedback_shift_engine<hydra_thrust::detail::uint32_t, 32u, 29u,  2u,  4u>, 0,
linear_feedback_shift_engine<hydra_thrust::detail::uint32_t, 32u, 28u,  3u, 17u>, 0
>,
0
> taus88;


typedef minstd_rand default_random_engine;



} 




using random::ranlux24;
using random::ranlux48;
using random::taus88;
using random::default_random_engine;

} 

