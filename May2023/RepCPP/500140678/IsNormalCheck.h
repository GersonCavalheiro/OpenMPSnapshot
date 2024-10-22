#pragma once
#include "CommonIncl.h"

inline bool should_check_normal(rr_uint itimestep) {
return itimestep % params.normal_check_step == 0;
}


bool check_finite(
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray<rr_float>& p,	
const heap_darray<rr_int>& itype,	
const rr_uint ntotal);


bool check_particles_are_within_boundaries(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray<rr_int>& itype);