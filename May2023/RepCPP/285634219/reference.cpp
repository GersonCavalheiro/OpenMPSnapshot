
#include "utils.hpp"

void commutator_reference(complex_t* sigma_in, complex_t* sigma_out, 
complex_t* hamiltonian, size_t dim, size_t num_sigma)
{
const size_t size_sigma = dim * dim;

#pragma omp parallel for
for (size_t n = 0; n < num_sigma; ++n)
{
size_t sigma_id = n * size_sigma;
#pragma novector
for (size_t i = 0; i < dim; ++i)
{
#pragma novector
for (size_t j = 0; j < dim; ++j)
{
complex_t tmp = 0.0; 
#pragma novector
for (size_t k = 0; k < dim; ++k)
{
tmp += hamiltonian[i * dim + k] * sigma_in[sigma_id + k * dim + j]
- sigma_in[sigma_id + i * dim + k] * hamiltonian[k * dim + j];
}
sigma_out[sigma_id + i * dim + j] -= complex_t(0.0,1.0) * hdt * tmp;
}
}
}
}

