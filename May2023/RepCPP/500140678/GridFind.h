#pragma once
#include "CommonIncl.h"

void grid_find(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
heap_darray_md<rr_uint>& neighbours); 

void find_neighbours(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,
const heap_darray<rr_uint>& grid,
const heap_darray<rr_uint>& cell_starts_in_grid,
heap_darray_md<rr_uint>& neighbours); 

void make_grid(
const rr_uint ntotal,
const heap_darray<rr_float2>& r,	
heap_darray<rr_uint>& grid,
heap_darray<rr_uint>& cells_start_in_grid); 