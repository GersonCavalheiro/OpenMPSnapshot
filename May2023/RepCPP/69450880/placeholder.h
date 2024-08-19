

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/actor.h>
#include <hydra/detail/external/hydra_thrust/detail/functional/argument.h>

namespace hydra_thrust
{
namespace detail
{
namespace functional
{

template<unsigned int i>
struct placeholder
{
typedef actor<argument<i> > type;
};

} 
} 
} 

