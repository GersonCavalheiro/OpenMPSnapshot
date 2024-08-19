#pragma once
#include "CommonIncl.h"


void int_force(
const rr_uint ntotal, 
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a);	

void find_stress_tensor(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
heap_darray<rr_float>& txx,
heap_darray<rr_float>& txy,
heap_darray<rr_float>& tyy);

void update_internal_state(
const rr_uint ntotal,
const heap_darray<rr_float>& rho,	
heap_darray<rr_float>& p);	

void find_internal_changes_pij_d_rhoij(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& txx,
const heap_darray<rr_float>& txy,
const heap_darray<rr_float>& tyy,
const heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a);	

void find_internal_changes_pidrho2i_pjdrho2j(
const rr_uint ntotal, 
const heap_darray<rr_float2>& v,	
const heap_darray<rr_float>& rho,	
const heap_darray_md<rr_uint>& neighbours, 
const heap_darray_md<rr_float2>& dwdr, 
const heap_darray<rr_float>& txx,
const heap_darray<rr_float>& txy,
const heap_darray<rr_float>& tyy,
const heap_darray<rr_float>& p,	
heap_darray<rr_float2>& a);	
