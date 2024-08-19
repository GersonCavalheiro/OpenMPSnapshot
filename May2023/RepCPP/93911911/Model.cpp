
#include "Model.h"

void edge_v::models::Model::getElAve( unsigned short         i_nElVes,
t_idx                  i_nEls,
t_idx          const * i_elVe,
float          const * i_velVe,
float                * o_velEl ) {
#ifdef PP_USE_OMP
#pragma omp parallel for
#endif
for( t_idx l_el = 0; l_el < i_nEls; l_el++ ) {
o_velEl[l_el] = 0;

for( unsigned short l_ve = 0; l_ve < i_nElVes; l_ve++ ) {
t_idx l_veId = i_elVe[l_el*i_nElVes + l_ve];
o_velEl[l_el] += i_velVe[l_veId];
}
o_velEl[l_el] /= i_nElVes;
}
}