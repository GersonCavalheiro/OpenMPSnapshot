#pragma once

#include "backend/tensor_traits.h"
#include "backend/tensor_traits_std.h"
#include "backend/tensor_traits_thrust.h"
#include "backend/tensor_traits_cusp.h"
#include "backend/blas2_dispatch_scalar.h"
#include "backend/blas2_dispatch_shared.h"
#include "backend/blas2_cusp.h"
#include "backend/blas2_sparseblockmat.h"
#include "backend/blas2_selfmade.h"
#include "backend/blas2_densematrix.h"
#ifdef MPI_VERSION
#include "backend/blas2_dispatch_mpi.h"
#endif 
#include "backend/blas2_dispatch_vector.h"



namespace dg{

namespace blas2{

namespace detail{

template< class ContainerType1, class MatrixType, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
static_assert( all_true<
dg::is_vector<ContainerType1>::value,
dg::is_vector<MatrixType>::value,
dg::is_vector<ContainerType2>::value>::value,
"The container types must have a vector data layout (AnyVector)!");
using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
using vector_category  = get_tensor_category<vector_type>;
static_assert( all_true<
dg::is_scalar_or_same_base_category<ContainerType1, vector_category>::value,
dg::is_scalar_or_same_base_category<ContainerType2, vector_category>::value
>::value,
"All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
return doDot_superacc( x, m, y, get_tensor_category<MatrixType>(), vector_category());
}

}


template< class ContainerType1, class MatrixType, class ContainerType2>
inline get_value_type<MatrixType> dot( const ContainerType1& x, const MatrixType& m, const ContainerType2& y)
{
std::vector<int64_t> acc = dg::blas2::detail::doDot_superacc( x,m,y);
return exblas::cpu::Round(acc.data());
}


template< class MatrixType, class ContainerType>
inline get_value_type<MatrixType> dot( const MatrixType& m, const ContainerType& x)
{
return dg::blas2::dot( x, m, x);
}
namespace detail{
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y,
AnyScalarTag)
{
dg::blas1::pointwiseDot( alpha, std::forward<MatrixType>(M), x, beta, y);
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y,
AnyScalarTag)
{
dg::blas1::pointwiseDot( std::forward<MatrixType>(M), x, y);
}

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y,
AnyMatrixTag)
{
static_assert( std::is_same<get_execution_policy<ContainerType1>,
get_execution_policy<ContainerType2>>::value,
"Vector types must have same execution policy");
static_assert( std::is_same<get_value_type<ContainerType1>,
get_value_type<MatrixType>>::value &&
std::is_same<get_value_type<ContainerType2>,
get_value_type<MatrixType>>::value,
"Vector and Matrix types must have same value type");
static_assert( std::is_same<get_tensor_category<ContainerType1>,
get_tensor_category<ContainerType2>>::value,
"Vector types must have same data layout");
dg::blas2::detail::doSymv( alpha, std::forward<MatrixType>(M), x, beta, y,
get_tensor_category<MatrixType>(),
get_tensor_category<ContainerType1>());
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y,
AnyMatrixTag)
{
static_assert( std::is_same<get_execution_policy<ContainerType1>,
get_execution_policy<ContainerType2>>::value,
"Vector types must have same execution policy");
static_assert( std::is_same<get_value_type<ContainerType1>,
get_value_type<MatrixType>>::value &&
std::is_same<get_value_type<ContainerType2>,
get_value_type<MatrixType>>::value,
"Vector and Matrix types must have same value type");
static_assert( std::is_same<get_tensor_category<ContainerType1>,
get_tensor_category<ContainerType2>>::value,
"Vector types must have same data layout");
dg::blas2::detail::doSymv( std::forward<MatrixType>(M), x, y,
get_tensor_category<MatrixType>(),
get_tensor_category<ContainerType1>());
}
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void doStencil(
FunctorType f,
MatrixType&& M,
const ContainerType1& x,
ContainerType2& y,
AnyMatrixTag)
{
static_assert( std::is_same<get_execution_policy<ContainerType1>,
get_execution_policy<ContainerType2>>::value,
"Vector types must have same execution policy");
static_assert( std::is_same<get_value_type<ContainerType1>,
get_value_type<MatrixType>>::value &&
std::is_same<get_value_type<ContainerType2>,
get_value_type<MatrixType>>::value,
"Vector and Matrix types must have same value type");
static_assert( std::is_same<get_tensor_category<ContainerType1>,
get_tensor_category<ContainerType2>>::value,
"Vector types must have same data layout");
dg::blas2::detail::doStencil( f, std::forward<MatrixType>(M), x, y,
get_tensor_category<MatrixType>(),
get_tensor_category<ContainerType1>());
}

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y,
NotATensorTag)
{
M(alpha,x,beta,y);
}
template< class MatrixType, class ContainerType1, class ContainerType2>
inline void doSymv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y,
NotATensorTag)
{
M(x,y);
}

}


template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y)
{
if(alpha == (get_value_type<ContainerType1>)0) {
dg::blas1::scal( y, beta);
return;
}
dg::blas2::detail::doSymv( alpha, std::forward<MatrixType>(M), x, beta, y, get_tensor_category<MatrixType>());
}




template< class MatrixType, class ContainerType1, class ContainerType2>
inline void symv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y)
{
dg::blas2::detail::doSymv( std::forward<MatrixType>(M), x, y, get_tensor_category<MatrixType>());
}

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void gemv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y)
{
dg::blas2::symv( alpha, std::forward<MatrixType>(M), x, beta, y);
}


template< class MatrixType, class ContainerType1, class ContainerType2>
inline void gemv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y)
{
dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}


template< class Stencil, class ContainerType, class ...ContainerTypes>
inline void parallel_for( Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs)
{
static_assert( all_true<
dg::is_vector<ContainerType>::value,
dg::is_vector<ContainerTypes>::value...>::value,
"All container types must have a vector data layout (AnyVector)!");
using vector_type = find_if_t<dg::is_not_scalar, ContainerType, ContainerType, ContainerTypes...>;
using tensor_category  = get_tensor_category<vector_type>;
static_assert( all_true<
dg::is_scalar_or_same_base_category<ContainerType, tensor_category>::value,
dg::is_scalar_or_same_base_category<ContainerTypes, tensor_category>::value...
>::value,
"All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
dg::blas2::detail::doParallelFor(tensor_category(), f, N, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
}

template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
inline void stencil(
FunctorType f,
MatrixType&& M,
const ContainerType1& x,
ContainerType2& y)
{
dg::blas2::detail::doStencil( f, std::forward<MatrixType>(M), x, y, get_tensor_category<MatrixType>());
}

template<class MatrixType, class AnotherMatrixType>
inline void transfer( const MatrixType& x, AnotherMatrixType& y)
{
dg::blas2::detail::doTransfer( x,y,
get_tensor_category<MatrixType>(),
get_tensor_category<AnotherMatrixType>());
}

} 

template< class MatrixType, class ContainerType1, class ContainerType2>
inline void apply( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y)
{
dg::blas2::symv( alpha, std::forward<MatrixType>(M), x, beta, y);
}


template< class MatrixType, class ContainerType1, class ContainerType2>
inline void apply( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y)
{
dg::blas2::symv( std::forward<MatrixType>(M), x, y);
}
} 
