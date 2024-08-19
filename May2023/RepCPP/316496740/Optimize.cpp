
#include "Optimize.h"


void optimize_center(const std::vector<std::array<float, Nv>>& Vec, std::vector<std::array<float, Nv>>& new_Center, const std::array<std::vector<int>, Nc>& Classes)
{
#pragma omp parallel for num_threads(NUM_THR) schedule(dynamic, 10)
for (int i = 0; i < Nc; i += 1)                                                                
{
std::array<float, Nv> Element = { 0.0 };                                               
for (int j = 0; j < Classes[i].size(); j += 1)                                         
{
#pragma omp simd
for (int k = 0; k < Nv; k += 1)                                                
{
Element[k] += (float)(Vec.at(Classes[i].at(j))[k] / Classes[i].size());
}
}
new_Center.at(i) = Element;                                                            
}
}
