#ifndef AMGCL_BACKEND_DETAIL_MATRIX_OPS_HPP
#define AMGCL_BACKEND_DETAIL_MATRIX_OPS_HPP





#include <type_traits>
#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>

namespace amgcl {
namespace backend {
namespace detail {

template <class Matrix, class Enable = void>
struct use_builtin_matrix_ops : std::false_type {};

} 

template <class Alpha, class Matrix, class Vector1, class Beta, class Vector2>
struct spmv_impl<
Alpha, Matrix, Vector1, Beta, Vector2,
typename std::enable_if<
detail::use_builtin_matrix_ops<Matrix>::value &&
math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector1>::type>::value &&
math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector2>::type>::value
>::type
>
{
static void apply(
Alpha alpha, const Matrix &A, const Vector1 &x, Beta beta, Vector2 &y
)
{
typedef typename value_type<Vector2>::type V;

const ptrdiff_t n = static_cast<ptrdiff_t>( rows(A) );

if (!math::is_zero(beta)) {
#pragma omp parallel for
for(ptrdiff_t i = 0; i < n; ++i) {
V sum = math::zero<V>();
for(typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
sum += a.value() * x[ a.col() ];
y[i] = alpha * sum + beta * y[i];
}
} else {
#pragma omp parallel for
for(ptrdiff_t i = 0; i < n; ++i) {
V sum = math::zero<V>();
for(typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
sum += a.value() * x[ a.col() ];
y[i] = alpha * sum;
}
}
}
};

template <class Matrix, class Vector1, class Vector2, class Vector3>
struct residual_impl<
Matrix, Vector1, Vector2, Vector3,
typename std::enable_if<
detail::use_builtin_matrix_ops<Matrix>::value &&
math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector1>::type>::value &&
math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector2>::type>::value &&
math::static_rows<typename value_type<Matrix>::type>::value == math::static_rows<typename value_type<Vector3>::type>::value
>::type
>
{
static void apply(
Vector1 const &rhs,
Matrix  const &A,
Vector2 const &x,
Vector3       &res
)
{
typedef typename value_type<Vector3>::type V;

const ptrdiff_t n = static_cast<ptrdiff_t>( rows(A) );

#pragma omp parallel for
for(ptrdiff_t i = 0; i < n; ++i) {
V sum = math::zero<V>();
for(typename row_iterator<Matrix>::type a = row_begin(A, i); a; ++a)
sum += a.value() * x[ a.col() ];
res[i] = rhs[i] - sum;
}
}
};


template <class Alpha, class Matrix, class Vector1, class Beta, class Vector2>
struct spmv_impl<
Alpha, Matrix, Vector1, Beta, Vector2,
typename std::enable_if<
detail::use_builtin_matrix_ops<Matrix>::value && (
math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector1>::type>::value ||
math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector2>::type>::value)
>::type
>
{
static void apply(
Alpha alpha, const Matrix &A, const Vector1 &x, Beta beta, Vector2 &y
)
{
typedef typename value_type<Matrix>::type V;

auto X = backend::reinterpret_as_rhs<V>(x);
auto Y = backend::reinterpret_as_rhs<V>(y);

spmv(alpha, A, X, beta, Y);
}
};

template <class Matrix, class Vector1, class Vector2, class Vector3>
struct residual_impl<
Matrix, Vector1, Vector2, Vector3,
typename std::enable_if<
detail::use_builtin_matrix_ops<Matrix>::value && (
math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector1>::type>::value ||
math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector2>::type>::value ||
math::static_rows<typename value_type<Matrix>::type>::value != math::static_rows<typename value_type<Vector3>::type>::value)
>::type
>
{
static void apply(
Vector1 const &f,
Matrix  const &A,
Vector2 const &x,
Vector3       &r
)
{
typedef typename value_type<Matrix>::type V;

auto X = backend::reinterpret_as_rhs<V>(x);
auto F = backend::reinterpret_as_rhs<V>(f);
auto R = backend::reinterpret_as_rhs<V>(r);

residual(F, A, X, R);
}
};

} 
} 

#endif
