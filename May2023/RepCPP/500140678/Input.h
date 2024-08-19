#pragma once
#include "CommonIncl.h"

void input(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_int>& itype,	
rr_uint& ntotal, 
rr_uint& nfluid, 
bool load_default_params = true); 

void fileInput(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_int>& itype,
rr_uint& ntotal, 
rr_uint& nfluid, 
std::string particles_path,
std::string params_path = "");

void cli(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p,	
heap_darray<rr_int>& itype,	
rr_uint& ntotal, 
rr_uint& nfluid); 

void loadDefaultParams();