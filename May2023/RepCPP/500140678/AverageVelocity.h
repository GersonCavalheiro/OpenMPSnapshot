#pragma once
#include "CommonIncl.h"


void average_velocity(
const rr_uint nfluid, 
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
heap_darray<rr_float2>& av); 