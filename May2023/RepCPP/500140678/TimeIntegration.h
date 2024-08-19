#pragma once
#include "CommonIncl.h"


void time_integration(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_int>& itype, 
const rr_uint start_ntotal, 
const rr_uint nfluid 
);

void predict_half_step(
const rr_uint ntotal,
const heap_darray<rr_int>& itype, 
const heap_darray<rr_float>& rho, 
const heap_darray<rr_float>& drho,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float2>& a,	
heap_darray<rr_float>& rho_predict, 
heap_darray<rr_float2>& v_predict); 

void whole_step(
const rr_uint ntotal,
const rr_uint timestep,
const heap_darray<rr_int>& itype, 
const heap_darray<rr_float>& drho,	
const heap_darray<rr_float2>& a,	
const heap_darray<rr_float2>& av,	
heap_darray<rr_float>& rho, 
heap_darray<rr_float2>& v,	
heap_darray<rr_float2>& r); 