#pragma once
#include "CommonIncl.h"

void update_acceleration(
const rr_uint nfluid, 
const rr_uint ntotal, 
const heap_darray<rr_int>& itype,	
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a,	
heap_darray<rr_float>& drho,	
heap_darray<rr_float2>& av); 

void update_change_rate(rr_uint nfluid,
const heap_darray<rr_float2>& indvxdt,
const heap_darray<rr_float2>& exdvxdt,
const heap_darray<rr_float2>& arvdvxdt,
heap_darray<rr_float2>& a);