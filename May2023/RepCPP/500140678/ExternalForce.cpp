#include "CommonIncl.h"

void update_repulsive_force_part(rr_uint ntotal,
rr_uint fluid_particle_idx,
const heap_darray<rr_float2>& r,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray<rr_int>& itype,	
heap_darray<rr_float2>& a) 
{
const rr_float rr0 = params.hsml;
const rr_float D = 5 * params.g * params.depth;
constexpr rr_uint p1 = 12;
constexpr rr_uint p2 = 4;

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, fluid_particle_idx), i != ntotal; 
++n)
{
if (itype(i) < 0) {

rr_float2 dr = r(fluid_particle_idx) - r(i);
rr_float rr = length(dr);

if (rr < rr0) {
rr_float f = D * (powun(rr0 / rr, p1) - powun(rr0 / rr, p2)) / sqr(rr);

a(fluid_particle_idx) += dr * f;
}
}
}
}

void external_force(
const rr_uint ntotal, 
const heap_darray<rr_float2>& r,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray<rr_int>& itype,	
heap_darray<rr_float2>& a) 
{
printlog_debug(__func__)();

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
a(j).x = 0;
a(j).y = -params.g;

if (params.sbt == 1 && itype(j) > 0) {
update_repulsive_force_part(ntotal, j,
r, neighbours, itype,
a);
}
}
}