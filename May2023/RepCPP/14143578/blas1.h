#pragma once

#include "backend/predicate.h"
#include "backend/tensor_traits.h"
#include "backend/tensor_traits_scalar.h"
#include "backend/tensor_traits_thrust.h"
#include "backend/tensor_traits_cusp.h"
#include "backend/tensor_traits_std.h"
#include "backend/blas1_dispatch_scalar.h"
#include "backend/blas1_dispatch_shared.h"
#include "backend/tensor_traits_cusp.h"
#ifdef MPI_VERSION
#include "backend/mpi_vector.h"
#include "backend/blas1_dispatch_mpi.h"
#endif
#include "backend/blas1_dispatch_vector.h"
#include "backend/blas1_dispatch_map.h"
#include "subroutines.h"



namespace dg{


namespace blas1
{
template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void evaluate( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs);






template< class ContainerType1, class ContainerType2>
inline get_value_type<ContainerType1> dot( const ContainerType1& x, const ContainerType2& y)
{
std::vector<int64_t> acc = dg::blas1::detail::doDot_superacc( x,y);
return exblas::cpu::Round(acc.data());
}


template< class ContainerType, class OutputType, class BinaryOp, class UnaryOp
= IDENTITY>
inline OutputType reduce( const ContainerType& x, OutputType zero, BinaryOp
binary_op, UnaryOp unary_op = UnaryOp())
{
return dg::blas1::detail::doReduce(
dg::get_tensor_category<ContainerType>(), x, zero, binary_op,
unary_op);
}


template<class ContainerTypeIn, class ContainerTypeOut>
inline void copy( const ContainerTypeIn& source, ContainerTypeOut& target){
if( std::is_same<ContainerTypeIn, ContainerTypeOut>::value && &source==(const ContainerTypeIn*)&target)
return;
dg::blas1::subroutine( dg::equals(), source, target);
}


template< class ContainerType>
inline void scal( ContainerType& x, get_value_type<ContainerType> alpha)
{
if( alpha == get_value_type<ContainerType>(1))
return;
dg::blas1::subroutine( dg::Scal<get_value_type<ContainerType>>(alpha), x );
}


template< class ContainerType>
inline void plus( ContainerType& x, get_value_type<ContainerType> alpha)
{
if( alpha == get_value_type<ContainerType>(0))
return;
dg::blas1::subroutine( dg::Plus<get_value_type<ContainerType>>(alpha), x );
}


template< class ContainerType, class ContainerType1>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, ContainerType& y)
{
using value_type = get_value_type<ContainerType>;
if( alpha == value_type(0) ) {
scal( y, beta);
return;
}
if( std::is_same<ContainerType, ContainerType1>::value && &x==(const ContainerType1*)&y){
dg::blas1::scal( y, (alpha+beta));
return;
}
dg::blas1::subroutine( dg::Axpby<get_value_type<ContainerType>>(alpha, beta),  x, y);
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void axpbypgz( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, const ContainerType2& y, get_value_type<ContainerType> gamma, ContainerType& z)
{
using value_type = get_value_type<ContainerType>;
if( alpha == value_type(0) )
{
axpby( beta, y, gamma, z);
return;
}
else if( beta == value_type(0) )
{
axpby( alpha, x, gamma, z);
return;
}
if( std::is_same<ContainerType1, ContainerType2>::value && &x==(const ContainerType1*)&y){
dg::blas1::axpby( alpha+beta, x, gamma, z);
return;
}
else if( std::is_same<ContainerType1, ContainerType>::value && &x==(const ContainerType1*)&z){
dg::blas1::axpby( beta, y, alpha+gamma, z);
return;
}
else if( std::is_same<ContainerType2, ContainerType>::value && &y==(const ContainerType2*)&z){
dg::blas1::axpby( alpha, x, beta+gamma, z);
return;
}
dg::blas1::subroutine( dg::Axpbypgz<get_value_type<ContainerType>>(alpha, beta, gamma),  x, y, z);
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void axpby( get_value_type<ContainerType> alpha, const ContainerType1& x, get_value_type<ContainerType> beta, const ContainerType2& y, ContainerType& z)
{
dg::blas1::evaluate( z , dg::equals(), dg::PairSum(), alpha, x, beta, y);
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
if( alpha == get_value_type<ContainerType>(0) ) {
dg::blas1::scal(y, beta);
return;
}
if( std::is_same<ContainerType, ContainerType1>::value && &x1==(const ContainerType1*)&y){
dg::blas1::subroutine( dg::AxyPby<get_value_type<ContainerType>>(alpha,beta), x2, y );

return;
}
if( std::is_same<ContainerType, ContainerType2>::value && &x2==(const ContainerType2*)&y){
dg::blas1::subroutine( dg::AxyPby<get_value_type<ContainerType>>(alpha,beta), x1, y );

return;
}
dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha,beta), x1, x2, y );
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDot( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
dg::blas1::evaluate( y, dg::equals(), dg::PairSum(), x1,x2);
}


template< class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3>
inline void pointwiseDot( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, const ContainerType3& x3, get_value_type<ContainerType> beta, ContainerType& y)
{
if( alpha == get_value_type<ContainerType>(0) ) {
dg::blas1::scal(y, beta);
return;
}
dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha,beta), x1, x2, x3, y );
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& x2, get_value_type<ContainerType> beta, ContainerType& y)
{
if( alpha == get_value_type<ContainerType>(0) ) {
dg::blas1::scal(y, beta);
return;
}
if( std::is_same<ContainerType, ContainerType1>::value && &x1==(const ContainerType1*)&y){
dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(alpha,beta), x2, y );

return;
}
dg::blas1::subroutine( dg::PointwiseDivide<get_value_type<ContainerType>>(alpha, beta), x1, x2, y );
}


template< class ContainerType, class ContainerType1, class ContainerType2>
inline void pointwiseDivide( const ContainerType1& x1, const ContainerType2& x2, ContainerType& y)
{
dg::blas1::evaluate( y, dg::equals(), dg::divides(), x1, x2);
}


template<class ContainerType, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void pointwiseDot(  get_value_type<ContainerType> alpha, const ContainerType1& x1, const ContainerType2& y1,
get_value_type<ContainerType> beta,  const ContainerType3& x2, const ContainerType4& y2,
get_value_type<ContainerType> gamma, ContainerType & z)
{
using value_type = get_value_type<ContainerType>;
if( alpha==value_type(0)){
pointwiseDot( beta, x2,y2, gamma, z);
return;
}
else if( beta==value_type(0)){
pointwiseDot( alpha, x1,y1, gamma, z);
return;
}
dg::blas1::subroutine( dg::PointwiseDot<get_value_type<ContainerType>>(alpha, beta, gamma), x1, y1, x2, y2, z );
}


template< class ContainerType, class ContainerType1, class UnaryOp>
inline void transform( const ContainerType1& x, ContainerType& y, UnaryOp op )
{
dg::blas1::subroutine( dg::Evaluate<dg::equals, UnaryOp>(dg::equals(),op), y, x);
}


template< class ContainerType, class BinarySubroutine, class Functor, class ContainerType0, class ...ContainerTypes>
inline void evaluate( ContainerType& y, BinarySubroutine f, Functor g, const ContainerType0& x0, const ContainerTypes& ...xs)
{
dg::blas1::subroutine( dg::Evaluate<BinarySubroutine, Functor>(f,g), y, x0, xs...);
}


namespace detail{

template< class ContainerType1, class ContainerType2>
inline std::vector<int64_t> doDot_superacc( const ContainerType1& x, const ContainerType2& y)
{
static_assert( all_true<
dg::is_vector<ContainerType1>::value,
dg::is_vector<ContainerType2>::value>::value,
"All container types must have a vector data layout (AnyVector)!");
using vector_type = find_if_t<dg::is_not_scalar, ContainerType1, ContainerType1, ContainerType2>;
using tensor_category  = get_tensor_category<vector_type>;
static_assert( all_true<
dg::is_scalar_or_same_base_category<ContainerType1, tensor_category>::value,
dg::is_scalar_or_same_base_category<ContainerType2, tensor_category>::value
>::value,
"All container types must be either Scalar or have compatible Vector categories (AnyVector or Same base class)!");
return doDot_superacc( x, y, tensor_category());
}

}


template< class Subroutine, class ContainerType, class ...ContainerTypes>
inline void subroutine( Subroutine f, ContainerType&& x, ContainerTypes&&... xs)
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
dg::blas1::detail::doSubroutine(tensor_category(), f, std::forward<ContainerType>(x), std::forward<ContainerTypes>(xs)...);
}

}


template<class from_ContainerType, class ContainerType, class ...Params>
inline void assign( const from_ContainerType& from, ContainerType& to, Params&& ... ps)
{
dg::detail::doAssign<from_ContainerType, ContainerType, Params...>( from, to, get_tensor_category<from_ContainerType>(), get_tensor_category<ContainerType>(), std::forward<Params>(ps)...);
}


template<class ContainerType, class from_ContainerType, class ...Params>
inline ContainerType construct( const from_ContainerType& from, Params&& ... ps)
{
return dg::detail::doConstruct<ContainerType, from_ContainerType, Params...>( from, get_tensor_category<ContainerType>(), get_tensor_category<from_ContainerType>(), std::forward<Params>(ps)...);
}


} 

