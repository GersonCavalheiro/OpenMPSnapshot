

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


__hydra_thrust_exec_check_disable__
template<typename BidirectionalIterator1,
typename BidirectionalIterator2>
__host__ __device__
BidirectionalIterator2 copy_backward(BidirectionalIterator1 first, 
BidirectionalIterator1 last, 
BidirectionalIterator2 result)
{
while (first != last)
{
--last;
--result;
*result = *last;
}

return result;
}


} 
} 
} 
} 

