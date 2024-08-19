#pragma once

#include <cstdint> 

#include "status.hxx" 

namespace sigma_config {

typedef struct {
double  occ[32][2]; 
int8_t  csv[32]; 
double  Z; 
double  rcut; 
double  sigma; 
uint8_t nn[8]; 
char    method[15]; 
int8_t  numax; 
} element_t;

element_t const & get(
double const Zcore 
, int const echo=0 
, char const **configuration=nullptr 
); 

void set_default_core_shells(int ncmx[4], double const Z); 

status_t all_tests(int const echo=0); 

} 
