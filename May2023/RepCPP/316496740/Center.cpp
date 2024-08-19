
#include "Center.h"


void init_centers(std::vector<std::array<float, Nv>>& old_Center)
{
std::random_device rd_Center;                                            
std::mt19937 mt_Center(rd_Center());                                     
std::uniform_real_distribution<float> dist_Center(0.0, MAX_LIMIT + 0.0); 
for (int i = 0; i < Nc; i += 1)
{
std::array<float, Nv> Elements;                                  
#pragma omp simd
for (int j = 0; j < Nv; j += 1)
{
Elements[j] = dist_Center(mt_Center);                    
}
old_Center.emplace_back(Elements);                               
}
}
