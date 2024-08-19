#pragma once

#include "status.hxx" 

namespace exchange_correlation {

template <typename real_t>
real_t lda_PZ81_kernel(     
real_t const rho,       
real_t &Vdn,            
real_t const mag=0,     
real_t *Vup=nullptr);   

template <typename real_t>
real_t lda_PW91_kernel(     
real_t const rho,       
real_t &Vdn,            
real_t const mag=0,     
real_t *Vup=nullptr);   


char const default_LDA[] = "PW";

template <typename real_t> inline
real_t LDA_kernel(          
real_t const rho,       
real_t &Vdn,            
real_t const mag=0,     
real_t *Vup=nullptr
) {
return ('W' == default_LDA[1]) ? 
lda_PW91_kernel(rho, Vdn, mag, Vup): 
lda_PZ81_kernel(rho, Vdn, mag, Vup); 
} 

status_t all_tests(int const echo=0); 

} 
