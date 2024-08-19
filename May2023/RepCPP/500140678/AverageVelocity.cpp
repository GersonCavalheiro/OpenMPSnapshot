#include "CommonIncl.h"
#include "Kernel.h"

void average_velocity(
const rr_uint nfluid, 
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
heap_darray<rr_float2>& av) 
{
printlog_debug(__func__)();

#pragma omp parallel for
for (rr_iter j = 0; j < nfluid; ++j) { 
av(j) = { 0.f };

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != params.ntotal; 
++n)
{
rr_float2 dvx = v(i) - v(j);
av(j) += dvx * params.mass / (rho(i) + rho(j)) * w(n, j) * 2.f;
}

av(j) *= params.average_velocity_epsilon;
}
}
