#ifndef AMGCL_RELAXATION_SPAI0_HPP
#define AMGCL_RELAXATION_SPAI0_HPP





#include <memory>
#include <amgcl/backend/interface.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace relaxation {


template <class Backend>
struct spai0 {
typedef typename Backend::value_type      value_type;
typedef typename Backend::matrix_diagonal matrix_diagonal;

typedef typename math::scalar_of<value_type>::type scalar_type;
typedef amgcl::detail::empty_params params;

template <class Matrix>
spai0( const Matrix &A, const params &, const typename Backend::params &backend_prm)
{
const size_t n = rows(A);

auto m = std::make_shared< backend::numa_vector<value_type> >(n, false);

#pragma omp parallel for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
value_type  num = math::zero<value_type>();
scalar_type den = math::zero<scalar_type>();

for(auto a = backend::row_begin(A, i); a; ++a) {
value_type v = a.value();
scalar_type norm_v = math::norm(v);
den += norm_v * norm_v;
if (a.col() == i) num += v;
}

(*m)[i] = math::inverse(den) * num;
}

M = Backend::copy_vector(m, backend_prm);
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_pre(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp
) const
{
static const scalar_type one = math::identity<scalar_type>();
backend::residual(rhs, A, x, tmp);
backend::vmul(one, *M, tmp, one, x);
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_post(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp
) const
{
static const scalar_type one = math::identity<scalar_type>();
backend::residual(rhs, A, x, tmp);
backend::vmul(one, *M, tmp, one, x);
}

template <class Matrix, class VectorRHS, class VectorX>
void apply( const Matrix&, const VectorRHS &rhs, VectorX &x) const
{
backend::vmul(math::identity<scalar_type>(), *M, rhs, math::zero<scalar_type>(), x);
}

size_t bytes() const {
return backend::bytes(*M);
}

std::shared_ptr<matrix_diagonal> M;
};

} 
} 

#endif
