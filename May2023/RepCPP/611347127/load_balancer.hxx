#pragma once

#include <cstdint> 

#include "status.hxx" 

namespace load_balancer {

double get(
uint32_t const comm_size 
, int32_t  const comm_rank 
, uint32_t const nb[3] 
, int const echo=0 
, double rank_center[4]=nullptr 
, uint16_t *owner_rank=nullptr 
); 

status_t all_tests(int const echo=0); 

} 
