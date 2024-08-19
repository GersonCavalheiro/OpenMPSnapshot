#pragma once

#include "status.hxx" 

namespace self_consistency {

status_t init(int const echo=0, float const ion=0.f); 

status_t all_tests(int const echo=0); 

} 
