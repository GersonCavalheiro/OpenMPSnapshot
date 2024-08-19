#pragma once
#include "CommonIncl.h"

void artificial_viscosity(
const rr_uint ntotal,	
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float2>& a); 