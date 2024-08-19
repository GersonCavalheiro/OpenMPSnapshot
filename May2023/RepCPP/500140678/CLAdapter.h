#pragma once
#include "CommonIncl.h"

void cl_time_integration(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
const heap_darray<rr_int>& itype, 
const rr_uint ntotal, 
const rr_uint nfluid);  