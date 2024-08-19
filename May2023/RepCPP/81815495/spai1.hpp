#ifndef AMGCL_RELAXATION_SPAI1_HPP
#define AMGCL_RELAXATION_SPAI1_HPP





#include <vector>

#include <memory>
#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/util.hpp>
#include <amgcl/detail/qr.hpp>

namespace amgcl {
namespace relaxation {


template <class Backend>
struct spai1 {
typedef typename Backend::value_type value_type;
typedef typename Backend::vector     vector;

typedef typename math::scalar_of<value_type>::type scalar_type;

typedef amgcl::detail::empty_params params;

template <class Matrix>
spai1( const Matrix &A, const params &, const typename Backend::params &backend_prm)
{
typedef typename backend::value_type<Matrix>::type value_type;

const size_t n = backend::rows(A);
const size_t m = backend::cols(A);

auto Ainv = std::make_shared<Matrix>(A);

#pragma omp parallel
{
std::vector<ptrdiff_t> marker(m, -1);
std::vector<ptrdiff_t> I, J;
std::vector<value_type> B, ek;
amgcl::detail::QR<value_type> qr;

#pragma omp for
for(ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
ptrdiff_t row_beg = A.ptr[i];
ptrdiff_t row_end = A.ptr[i + 1];

I.assign(A.col + row_beg, A.col + row_end);

J.clear();
for(ptrdiff_t j = row_beg; j < row_end; ++j) {
ptrdiff_t c = A.col[j];

for(ptrdiff_t jj = A.ptr[c], ee = A.ptr[c + 1]; jj < ee; ++jj) {
ptrdiff_t cc = A.col[jj];
if (marker[cc] < 0) {
marker[cc] = 1;
J.push_back(cc);
}
}
}
std::sort(J.begin(), J.end());
B.assign(I.size() * J.size(), math::zero<value_type>());
ek.assign(J.size(), math::zero<value_type>());
for(size_t j = 0; j < J.size(); ++j) {
marker[J[j]] = j;
if (J[j] == static_cast<ptrdiff_t>(i)) ek[j] = math::identity<value_type>();
}

for(ptrdiff_t j = row_beg; j < row_end; ++j) {
ptrdiff_t c = A.col[j];

for(auto a = row_begin(A, c); a; ++a)
B[marker[a.col()] + J.size() * (j - row_beg)] = a.value();
}

qr.solve(J.size(), I.size(), &B[0], &ek[0], &Ainv->val[row_beg],
amgcl::detail::col_major);

for(size_t j = 0; j < J.size(); ++j)
marker[J[j]] = -1;
}
}

M = Backend::copy_matrix(Ainv, backend_prm);
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_pre(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp
) const
{
backend::residual(rhs, A, x, tmp);
backend::spmv(math::identity<scalar_type>(), *M, tmp, math::identity<scalar_type>(), x);
}

template <class Matrix, class VectorRHS, class VectorX, class VectorTMP>
void apply_post(
const Matrix &A, const VectorRHS &rhs, VectorX &x, VectorTMP &tmp
) const
{
backend::residual(rhs, A, x, tmp);
backend::spmv(math::identity<scalar_type>(), *M, tmp, math::identity<scalar_type>(), x);
}

template <class Matrix, class VectorRHS, class VectorX>
void apply(const Matrix&, const VectorRHS &rhs, VectorX &x) const
{
backend::spmv(math::identity<scalar_type>(), *M, rhs, math::zero<scalar_type>(), x);
}

size_t bytes() const {
return backend::bytes(*M);
}

std::shared_ptr<typename Backend::matrix> M;
};

} 

namespace backend {

template <class Backend>
struct relaxation_is_supported<
Backend, relaxation::spai1,
typename std::enable_if<
(amgcl::math::static_rows<typename Backend::value_type>::value > 1)
>::type
> : std::false_type
{};

} 
} 


#endif
