#pragma once
#include "CommonIncl.h"


void virt_part(
const rr_uint nfluid,
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_int>& itype); 

void dynamic_boundaries(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
const rr_float time);

rr_uint count_virt_part_num();