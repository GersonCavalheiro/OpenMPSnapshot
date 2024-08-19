#pragma once

#include <cstdint> 

#include "status.hxx" 

namespace single_atom {

status_t atom_update(
char const *what    
, int natoms          
, double  *dp=nullptr 
, int32_t *ip=nullptr 
, float   *fp=nullptr 
, double *const *dpp=nullptr 
); 

status_t all_tests(int const echo=0); 

} 
