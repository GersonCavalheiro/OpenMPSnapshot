#pragma once

void rzm_generator(
const heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& a,
const rr_uint nfluid,
const rr_float time);
void rzm_absorber(
const heap_darray<rr_float2>& r,	
const heap_darray<rr_float2>& v,	
heap_darray<rr_float2>& a,
const rr_uint nfluid,
const rr_float time);

void impulse_nwm(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& a,
const rr_uint nfluid,
const rr_uint ntotal,
const rr_float time);

void make_waves(
heap_darray<rr_float2>& r,	
heap_darray<rr_float2>& v,	
heap_darray<rr_float2>& a,
const rr_uint nfluid,
const rr_uint ntotal,
const rr_float time);