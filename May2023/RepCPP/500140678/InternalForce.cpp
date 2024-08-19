#include "CommonIncl.h"
#include "EOS.h"
#include "Kernel.h"
#include "InternalForce.h"

#include <iostream>

void find_stress_tensor(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float>& txx,
heap_darray<rr_float>& txy,
heap_darray<rr_float>& tyy)
{
printlog_debug(__func__)();
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
txx(j) = 0.f;
txy(j) = 0.f;
tyy(j) = 0.f;

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float dwdx = dwdr(n, j).x;
rr_float dwdy = dwdr(n, j).y;
rr_float2 dvx = v(j) - v(i);

rr_float hxx = 2.f * dvx.x * dwdx - dvx.y * dwdy;
rr_float hxy = dvx.x * dwdy + dvx.y * dwdx;
rr_float hyy = 2.f * dvx.y * dwdy - dvx.x * dwdx;
hxx *= 2.f / 3.f;
hyy *= 2.f / 3.f;

txx(j) += params.mass * hxx / rho(i);
txy(j) += params.mass * hxy / rho(i);
tyy(j) += params.mass * hyy / rho(i);
}
}
}


void find_internal_changes_pij_d_rhoij(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& txx,
const heap_darray<rr_float>& txy,
const heap_darray<rr_float>& tyy,
const heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a)	
{
printlog_debug(__func__)();
rr_float eta = params.water_dynamic_visc;
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
a(j) = { 0.f };

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 h = -dwdr(n, j) * (p(i) + p(j));
rr_float rhoij = 1.f / (rho(i) * rho(j));

if (params.visc) {
rr_float dwdx = dwdr(n, j).x;
rr_float dwdy = dwdr(n, j).y;

h.x += (txx(i) + txx(j)) * dwdx * eta;
h.x += (txy(i) + txy(j)) * dwdy * eta;
h.y += (txy(i) + txy(j)) * dwdx * eta;
h.y += (tyy(i) + tyy(j)) * dwdy * eta;
}

a(j) -= h * params.mass * rhoij;
}
}
}
void find_internal_changes_pidrho2i_pjdrho2j(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& txx,
const heap_darray<rr_float>& txy,
const heap_darray<rr_float>& tyy,
const heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a)	
{
printlog_debug(__func__)();
rr_float eta = params.water_dynamic_visc;
#pragma omp parallel for
for (rr_iter j = 0; j < ntotal; ++j) { 
a(j) = { 0.f };

rr_uint i;
for (rr_iter n = 0;
i = neighbours(n, j), i != ntotal; 
++n)
{
rr_float2 h = -dwdr(n, j) * (p(i) / sqr(rho(i)) + p(j) / sqr(rho(j)));

if (params.visc) { 
rr_float dwdx = dwdr(n, j).x;
rr_float dwdy = dwdr(n, j).y;

h.x += (txx(i) / sqr(rho(i)) + txx(j) / sqr(rho(j))) * dwdx * eta;
h.x += (txy(i) / sqr(rho(i)) + txy(j) / sqr(rho(j))) * dwdy * eta;
h.y += (txy(i) / sqr(rho(i)) + txy(j) / sqr(rho(j))) * dwdx * eta;
h.y += (tyy(i) / sqr(rho(i)) + tyy(j) / sqr(rho(j))) * dwdy * eta;
}

a(j) -= h * params.mass;
}
}
}

void update_internal_state(
const rr_uint ntotal,
const heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p)	
{
printlog_debug(__func__)();

for (rr_uint i = 0; i < ntotal; i++) {
p(i) = p_art_water(rho(i));
}
}


void int_force(
const rr_uint ntotal, 
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a)	
{
printlog_debug(__func__)();

static heap_darray<rr_float> txx(params.maxn);
static heap_darray<rr_float> tyy(params.maxn);
static heap_darray<rr_float> txy(params.maxn);

if (params.visc) {
find_stress_tensor(ntotal,
v, rho,
neighbours, dwdr,
txx, txy, tyy);
}

update_internal_state(ntotal,
rho,
p);

if (params.pa_sph == 1) {
find_internal_changes_pij_d_rhoij(ntotal,
v, rho,
neighbours, dwdr,
txx, txy, tyy,
p,
a);
}
else {
find_internal_changes_pidrho2i_pjdrho2j(ntotal,
v, rho,
neighbours, dwdr,
txx, txy, tyy,
p,
a);
}
}
