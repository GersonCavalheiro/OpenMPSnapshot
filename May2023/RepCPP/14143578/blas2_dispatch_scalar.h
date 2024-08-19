#pragma once
#include "blas1_dispatch_scalar.h"
#include "blas1_dispatch_shared.h"
#include "blas1_dispatch_vector.h"

namespace dg
{
namespace blas2
{
namespace detail
{

template< class ContainerType1, class MatrixType, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const MatrixType& m, const ContainerType2& y);

template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, AnyScalarTag, AnyScalarTag)
{
static_assert( std::is_convertible<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
static_assert( std::is_convertible<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
const get_value_type<Vector1>* x_ptr = &x;
const get_value_type<Matrix>* m_ptr = &m;
const get_value_type<Vector2>* y_ptr = &y;
std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
int status = 0;
exblas::exdot_cpu<const get_value_type<Vector1>*, const get_value_type<Matrix>*, const get_value_type<Vector2>*, 3>( 1, x_ptr,m_ptr,y_ptr, &h_superacc[0], &status) ;
if(status != 0)
throw dg::Error(dg::Message(_ping_)<<"Scalar Dot failed since one of the inputs contains NaN or Inf");
return h_superacc;
}
template< class Vector1, class Matrix, class Vector2 >
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, AnyScalarTag, SharedVectorTag)
{
static_assert( std::is_convertible<get_value_type<Vector1>, double>::value, "We only support double precision dot products at the moment!");
static_assert( std::is_convertible<get_value_type<Vector2>, double>::value, "We only support double precision dot products at the moment!");
using vector_type = find_if_t<dg::is_not_scalar, Vector1, Vector1, Vector2>;
constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
using execution_policy = get_execution_policy<vector_type>;
static_assert( all_true<
dg::has_any_or_same_policy<Vector1, execution_policy>::value,
dg::has_any_or_same_policy<Vector2, execution_policy>::value
>::value,
"All ContainerType types must have compatible execution policies (AnyPolicy or Same)!");
auto size = get_idx<vector_idx>(x,y).size();
return dg::blas1::detail::doDot_dispatch( execution_policy(), size, do_get_pointer_or_reference(x,get_tensor_category<Vector1>()), do_get_pointer_or_reference(m,get_tensor_category<Matrix>()), do_get_pointer_or_reference(y,get_tensor_category<Vector2>()));
}
template< class Vector1, class Matrix, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x, const Matrix& m, const Vector2& y, AnyScalarTag, RecursiveVectorTag)
{
constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
auto size = get_idx<vector_idx>(x,y).size();
std::vector<std::vector<int64_t>> acc( size);
for( unsigned i=0; i<size; i++)
acc[i] = doDot_superacc( do_get_vector_element(x,i,get_tensor_category<Vector1>()), m, do_get_vector_element(y,i,get_tensor_category<Vector2>()));
for( unsigned i=1; i<size; i++)
{
int imin = exblas::IMIN, imax = exblas::IMAX;
exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
imin = exblas::IMIN, imax = exblas::IMAX;
exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
acc[0][k] += acc[i][k];
}
return acc[0];
}
} 
} 
} 
