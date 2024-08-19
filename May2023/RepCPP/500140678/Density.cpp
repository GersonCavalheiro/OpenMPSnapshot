#include "CommonIncl.h" 
#include "Kernel.h"
#include "GridUtils.h"

static void density_normalization(
const rr_uint ntotal,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
const heap_darray<rr_float>& rho,	
heap_darray<rr_float>& normrho) 
{
printlog_debug(__func__)();

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
rr_float wjj = kernel_w(0, params.density_skf);
normrho(j) = params.mass / rho(j) * wjj;

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
normrho(j) += params.mass / rho(i) * w(n, j);
}
}
}
static void density_summation(
const rr_uint ntotal,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
heap_darray<rr_float>& rho)	
{
printlog_debug(__func__)();

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
rr_float wjj = kernel_w(0, params.density_skf);
rho(j) = params.mass * wjj;

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rho(j) += params.mass * w(n, j);
}
}
}
void sum_density(
const rr_uint ntotal,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
heap_darray<rr_float>& rho) 
{
printlog_debug(__func__)();

static heap_darray<rr_float> normrho(params.maxn);
if (params.nor_density) {
density_normalization(
ntotal,
neighbours,
w,
rho,
normrho);
}

density_summation(
ntotal,
neighbours,
w,
rho);

if (params.nor_density) {
for (rr_uint k = 0; k < ntotal; k++) {
rho(k) /= normrho(k);
}
}
}

void con_density(
const rr_uint ntotal,	
const heap_darray<rr_float2>& v,
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& rho,	
heap_darray<rr_float>& drhodt) 
{
printlog_debug(__func__)();

#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
drhodt(j) = 0.f;

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 dvx = v(i) - v(j);
rr_float vcc = dot(dvx, dwdr(n, j));
drhodt(j) += params.mass * vcc;
}
}
}