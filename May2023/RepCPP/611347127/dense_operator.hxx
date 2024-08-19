#pragma once

#include <cstdio> 

#include "status.hxx" 

#include "linear_algebra.hxx" 
#include "inline_math.hxx" 
#include "complex_tools.hxx" 


namespace dense_operator {

template <typename wave_function_t> 
class dense_operator_t {

public:
typedef wave_function_t complex_t;

private:
complex_t const *Hmt, *Smt; 
complex_t const *Cnd; 
int nB, nBa;

inline status_t matrix_vector_multiplication(complex_t mvec[]
, complex_t const mat[], complex_t const vec[], int const echo=0) const {
if (echo > 19) {
std::printf("# %s<%s> gemm\n", __func__, complex_name<complex_t>());
std::fflush(stdout);
} 
return linear_algebra::gemm(nB, 1, nB, mvec, nB, vec, nB, mat, nBa);
} 

public:

dense_operator_t(
int const nB                 
, int const stride             
, complex_t const *Hmt         
, complex_t const *Smt=nullptr 
, complex_t const *Cnd=nullptr 
)
: Hmt{Hmt}, Smt{Smt}, Cnd{Cnd}, nB{nB}, nBa{stride}
{
assert( nB <= nBa );
} 

status_t Hamiltonian(complex_t Hpsi[], complex_t const psi[], int const echo=0) const {
return matrix_vector_multiplication(Hpsi, Hmt, psi, echo); 
} 

status_t Overlapping(complex_t Spsi[], complex_t const psi[], int const echo=0) const {
return use_overlap() ? matrix_vector_multiplication(Spsi, Smt, psi, echo) : 0;
} 

status_t Conditioner(complex_t Cpsi[], complex_t const psi[], int const echo=0) const {
if (use_precond()) product(Cpsi, nB, Cnd, psi); 
return 0;
} 

double get_volume_element() const { return 1.0; }
size_t get_degrees_of_freedom() const { return size_t(nB); }
bool use_precond() const { return (nullptr != Cnd); }
bool use_overlap() const { return (nullptr != Smt); }
}; 

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

inline status_t test_construct_and_destroy(int const echo=0) {
double const matrix[3][4] = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}};
dense_operator_t<double> const op(3, 4, matrix[0]);
return op.get_degrees_of_freedom() - 3;
} 

inline status_t all_tests(int const echo=0) {
status_t stat(0);
stat += test_construct_and_destroy(echo);
return stat;
} 

#endif 

} 

#ifdef DEBUG
#undef DEBUG
#endif
