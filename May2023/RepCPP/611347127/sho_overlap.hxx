#pragma once

#include "status.hxx" 

namespace sho_overlap {

template <typename real_t>
status_t moment_tensor(
real_t tensor[] 
, double const distance
, int const n1
, int const n0
, double const sigma1=1.0
, double const sigma0=1.0
, int const maxmoment=0
); 

template <typename real_t> inline
status_t overlap_matrix(
real_t matrix[] 
, double const distance
, int const n1
, int const n0
, double const sigma1=1.0
, double const sigma0=1.0
) {
return moment_tensor(matrix, distance, n1, n0, sigma1, sigma0, 0);
} 

template <typename real_t> inline 
status_t nabla2_matrix(
real_t matrix[] 
, double const distance
, int const n1
, int const n0
, double const sigma1=1.0
, double const sigma0=1.0
) {
return moment_tensor(matrix, distance, n1, n0, sigma1, sigma0, -2);
} 

status_t moment_normalization(
double matrix[] 
, int const m
, double const sigma=1.0
, int const echo=0
); 

template <typename real_t>
status_t product_tensor(
real_t tensor[]
, int const n 
, double const sigma=2.0 
, double const sigma1=1.0
, double const sigma0=1.0
); 

status_t all_tests(int const echo=0); 

} 
