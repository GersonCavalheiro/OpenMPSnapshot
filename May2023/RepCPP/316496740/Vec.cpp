
#include "Vec.h"


void vec_init(std::vector<std::array<float, Nv>>& Vec)
{
std::random_device rd_Vec;                                            
std::mt19937 mt_Vec(rd_Vec());                                        
std::uniform_real_distribution<float> dist_Vec(0.0, MAX_LIMIT + 0.0); 
for (int i = 0; i < N; i += 1)
{
std::array<float, Nv> Elements;                               
#pragma omp simd
for (int j = 0; j < Nv; j += 1)
{
Elements[j] = dist_Vec(mt_Vec);                       
}
Vec.emplace_back(Elements);                                   
}
}
