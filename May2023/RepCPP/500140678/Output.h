#pragma once
#include "CommonIncl.h"
#include <optional>

void dump(
heap_darray<rr_float2>&& r,
heap_darray<rr_int>&& itype,
heap_darray<rr_float2>&& v,
heap_darray<rr_float>&& rho,
heap_darray<rr_float>&& p,
const rr_uint itimestep);

void output(
heap_darray<rr_float2>&& r,
heap_darray<rr_int>&& itype,
std::optional<heap_darray<rr_float2>> v,
std::optional<heap_darray<rr_float>> rho,
std::optional<heap_darray<rr_float>> p,
const rr_uint itimestep);

void fast_output(
heap_darray<rr_float2>&& r,	
const heap_darray<rr_int>& itype,	
const rr_uint ntotal,	
const rr_uint itimestep); 

void setupOutput();

void printParams();

void printTimeEstimate(long long totalTime_ns, rr_uint timeStep);

void makeParamsHeader(std::string path = "cl/clparams.h");