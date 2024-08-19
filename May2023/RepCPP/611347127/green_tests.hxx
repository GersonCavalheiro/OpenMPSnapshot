#pragma once


#include <vector> 
#include <string> 
#include <tuple> 

#include "status.hxx" 

namespace green_tests {

void add_tests(
std::vector<std::tuple<char const*, double, status_t>> & results 
, std::string const & input_name 
, bool const show 
, bool const all 
, int const echo 
); 

status_t all_tests(int const echo=0); 

} 
