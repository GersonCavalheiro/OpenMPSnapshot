#pragma once
#include "CommonIncl.h"

void sum_density(
const rr_uint ntotal,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float>& w, 
heap_darray<rr_float>& rho); 

void con_density(
const rr_uint ntotal,	
const heap_darray<rr_float2>& v,
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& rho,	
heap_darray<rr_float>& drhodt); 