#pragma once

#include "status.hxx" 
#include "sho_tools.hxx" 

namespace sho_unitary {

class Unitary_SHO_Transform {
public:

Unitary_SHO_Transform(int const lmax=7, int const echo=8); 

~Unitary_SHO_Transform() {
for (int nu = 0; nu <= numax_; ++nu) {
delete[] u_[nu];
} 
delete[] u_;
} 

double get_entry(int const nzyx, int const nlnm) const; 

status_t construct_dense_matrix(
double matrix[]
, int const nu_max
, int const matrix_stride=-1
, sho_tools::SHO_order_t const row_order=sho_tools::order_Ezyx 
, sho_tools::SHO_order_t const col_order=sho_tools::order_Elnm 
) const; 

status_t transform_vector(double out[], sho_tools::SHO_order_t const out_order
, double const inp[], sho_tools::SHO_order_t const inp_order
, int const nu_max, int const echo=0) const; 

double test_unitarity(int const echo=7) const; 

inline int numax() const { return numax_; }

private: 
double **u_; 
int numax_;  

}; 

status_t all_tests(int const echo=0); 

} 
