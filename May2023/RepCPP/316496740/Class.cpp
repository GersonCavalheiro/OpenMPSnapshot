
#include "Class.h"


void compute_classes(const std::vector<std::array<float, Nv>>& Vec, const std::vector<std::array<float, Nv>>& old_Center, std::array<std::vector<int>, Nc>& Classes)
{
#pragma omp parallel for num_threads(NUM_THR) schedule(dynamic, 10)
for (int i = 0; i < Nc; i += 1)                                                     
{
Classes[i].clear();                                                         
Classes[i].reserve((int)N / Nc);                                            
}
int argmin_idx = -1;                                                                
long double argmin_val = std::numeric_limits<long int>::max() + 0.0;                
#pragma omp parallel for num_threads(NUM_THR) firstprivate(argmin_idx, argmin_val) schedule(dynamic, 1000) 
for (int i = 0; i < N; i += 1)                                                      
{
for (int j = 0; j < Nc; j += 1)                                             
{
long double temp_eucl_dist = eucl_diff(Vec.at(i), old_Center.at(j));
if (argmin_val > temp_eucl_dist)                                    
{
argmin_val = temp_eucl_dist;                                
argmin_idx = j;                                             
}
}
#pragma omp critical
Classes[argmin_idx].emplace_back(i);
argmin_idx = -1;                                                            
argmin_val = std::numeric_limits<unsigned long int>::max();                 
}
}
