#include "CommonIncl.h"
#include "Kernel.h"
#include "EOS.h"

void artificial_viscosity(
const rr_uint ntotal,	
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float2>& a) 
{
printlog_debug(__func__)();
rr_float alpha = params.artificial_shear_visc;
rr_float beta = params.artificial_bulk_visc;
static constexpr rr_float etq = 0.1f;

static const rr_float c_ij = c_art_water();

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
a(j) = { 0.f };

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 dv = v(i) - v(j);
rr_float2 dr = r(i) - r(j);
rr_float vr = dot(dv, dr);
rr_float rr = length_sqr(dr);

if (vr < 0) {
rr_float muv = params.hsml * vr / (rr + sqr(params.hsml * etq));

rr_float mrho = 0.5f * (rho(i) + rho(j));
rr_float piv = (beta * muv - alpha * c_ij) * muv / mrho;

rr_float2 h = -dwdr(n, j) * piv;
a(j) -= h * params.mass;
}
}
}
}
