#pragma once
#include "CommonIncl.h"

void external_force(
const rr_uint ntotal, 
const heap_darray<rr_float2>& r,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray<rr_int>& itype,	
heap_darray<rr_float2>& a); 


void update_repulsive_force_part(rr_uint ntotal,
rr_uint fluid_particle_idx,
const heap_darray<rr_float2>& r,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray<rr_int>& itype,	
heap_darray<rr_float2>& a); 