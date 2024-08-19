#pragma once

#include <cstdio> 
#include <vector> 

#define CRTP_printf(...)

#include "vector_layout.hxx" 
#include "complex_tools.hxx" 
#include "status.hxx" 

template <class CRTP_t, typename real_t>
class LinOp {

public:

inline void apply(real_t Ax[], real_t const x[]) const {
CRTP_printf("# %s:%i %s<%s>\n", __FILE__, __LINE__, __func__, complex_name<real_t>());
return static_cast<CRTP_t const *>(this)->_apply(Ax, x);
} 

inline void axpby(real_t y[], real_t const x[], real_t const *a=nullptr, real_t const *b=nullptr) const {
CRTP_printf("# %s:%i %s<%s>\n", __FILE__, __LINE__, __func__, complex_name<real_t>());
return static_cast<CRTP_t const *>(this)->layout_.axpby(y, x, a, b);
} 

inline void inner(real_t d[], real_t const x[], real_t const *y=nullptr) const {
CRTP_printf("# %s:%i %s<%s>\n", __FILE__, __LINE__, __func__, complex_name<real_t>());
return static_cast<CRTP_t const *>(this)->layout_.inner(d, x, y);
} 

private:

}; 



namespace linear_operator {

#ifdef  NO_UNIT_TESTS
inline status_t all_tests(int const echo=0) { return STATUS_TEST_NOT_INCLUDED; }
#else 

template <typename real_t>
class DiagOp : public LinOp<DiagOp<real_t>,real_t> {
public:

DiagOp(VecLayout<real_t> const & layout) 
: layout_(layout) 
, diagonal_(layout.ndof(), 3.0) { 
CRTP_printf("# %s:%i %s<%s> (constructor)\n", __FILE__, __LINE__, __func__, complex_name<real_t>());
} 

void _apply(real_t Dx[], real_t const x[]) const {
CRTP_printf("# %s:%i %s<%s>\n", __FILE__, __LINE__, __func__, complex_name<real_t>());
auto const m = layout_.stride(); 
for (int i = 0; i < layout_.ndof(); ++i) {
for (int j = 0; j < layout_.nrhs(); ++j) {
auto const ij = i*m + j;
Dx[ij] = diagonal_[i]*x[ij]; 
} 
} 
} 

VecLayout<real_t> layout_;
private:
std::vector<real_t> diagonal_; 
}; 


inline status_t test_DiagOp(int const echo=4) {
if (echo > 0) std::printf("# %s\n", __func__);

typedef double real_t;
std::vector<real_t> Hx(7*8, 0.0), x(7*8, 1.0), xHx(8, 0.0);
VecLayout<real_t> layout(7, 8);
DiagOp<real_t> hamiltonian(layout);
hamiltonian.apply(Hx.data(), x.data());
hamiltonian.inner(xHx.data(), x.data(), Hx.data());
hamiltonian.axpby(x.data(), Hx.data(), xHx.data());

if (echo > 0) {
for (unsigned i = 0; i < xHx.size(); ++i) { 
std::printf(" %g", xHx[i]); 
}   std::printf("\n"); 
} 

return 0;
} 

inline status_t all_tests(int const echo=0) {
if (echo > 0) std::printf("# %s\n", __func__);
status_t stat = 0;
stat += test_DiagOp(echo);
return stat;
} 


#endif 

} 
