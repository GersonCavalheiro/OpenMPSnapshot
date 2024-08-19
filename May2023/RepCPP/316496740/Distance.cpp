
#include "Distance.h"


long double eucl_diff(const std::array<float, Nv>& A, const std::array<float, Nv>& B)
{
long double eucl_diff = 0;
#pragma omp parallel for simd num_threads(NUM_THR) reduction(+: eucl_diff) schedule(dynamic, CHUNK_SIZE)
for (int i = 0; i < Nv; i += 1)                                                  
{
long double point_diff = A[i] - B[i];                                    
if (point_diff > std::numeric_limits<float>::max())                      
{
point_diff = std::numeric_limits<float>::max();                  
}
long double point_diff_squared = point_diff * point_diff;                
eucl_diff += point_diff_squared;                                         
}
return eucl_diff;
}


long double convergence(const std::vector<std::array<float, Nv>>& curr_Center, const std::vector<std::array<float, Nv>>& prev_Center)
{
long double convergence_sum = 0;                                                 
for (int i = 0; i < Nc; i += 1)                                                  
{
long double tmp_eucl_d = eucl_diff(curr_Center.at(i), prev_Center.at(i));
convergence_sum += tmp_eucl_d > 1.0 ? tmp_eucl_d : 0.0;                  
if (convergence_sum > std::numeric_limits<double>::max())                
{
convergence_sum = std::numeric_limits<double>::max();            
}
}
return convergence_sum;
}


long double normalize_convergence(const long double curr_iter_conv, const long double prev_iter_conv, const int iter_counter)
{
return curr_iter_conv > Nc && iter_counter > 2 ? std::abs(curr_iter_conv - prev_iter_conv) / std::max(curr_iter_conv, prev_iter_conv) :
iter_counter <= 2 ? 1.0 : 0.0;
}
