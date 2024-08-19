#pragma once

#include <cmath> 
#include <vector> 

#include "status.hxx" 

namespace radial_r2grid {



template <typename real_t=double>
std::vector<real_t> r_axis(int const nr2, float const ar2=1) {
double const ar2inv = 1.0/ar2;
std::vector<real_t> r(nr2);
for (int ir2 = 0; ir2 < nr2; ++ir2) {
r[ir2] = std::sqrt(ir2*ar2inv); 
} 
return r;
} 


} 
