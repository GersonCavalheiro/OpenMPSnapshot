#pragma once

#include <cstdint> 

#include "status.hxx" 

namespace chemical_symbol {

int8_t decode(char const S, char const y); 
int8_t get(char Sy[4], double const Z, char const blank='\0'); 

status_t all_tests(int const echo=0); 

} 
