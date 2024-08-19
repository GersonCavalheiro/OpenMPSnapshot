#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#include <vector>
#include <array>
#include "blas1_dispatch_shared.h"
#include "vector_categories.h"
#include "tensor_traits.h"
#ifdef _OPENMP
#include <omp.h>
#endif 

namespace dg
{
template<class to_ContainerType, class from_ContainerType, class ...Params>
inline to_ContainerType construct( const from_ContainerType& src, Params&& ...ps);
template<class from_ContainerType, class to_ContainerType, class ...Params>
inline void assign( const from_ContainerType&, to_ContainerType&, Params&& ...ps);

namespace detail{

template<class To, class From, class ...Params>
To doConstruct( const From& src, ArrayVectorTag, SharedVectorTag, Params&&...ps )
{
To t;
using inner_vector = typename To::value_type;
for (unsigned i=0; i<t.size(); i++)
t[i] = dg::construct<inner_vector>(src, std::forward<Params>(ps)...);
return t;
}
template<class To, class From, class ...Params>
To doConstruct( const From& src, ArrayVectorTag, MPIVectorTag, Params&&...ps )
{
To t;
using inner_vector = typename To::value_type;
for (unsigned i=0; i<t.size(); i++)
t[i] = dg::construct<inner_vector>(src, std::forward<Params>(ps)...);
return t;
}
template<class To, class From, class ...Params>
To doConstruct( const From& src, ArrayVectorTag, RecursiveVectorTag, Params&&...ps )
{
To t;
using inner_vector = typename To::value_type;
for (unsigned i=0; i<t.size(); i++)
t[i] = dg::construct<inner_vector>(src[i], std::forward<Params>(ps)...);
return t;
}

template<class To, class From, class Size, class ...Params>
To doConstruct( const From& src, RecursiveVectorTag, SharedVectorTag, Size size, Params&&... ps )
{
To t(size);
using inner_vector = typename To::value_type;
for (int i=0; i<(int)size; i++)
t[i] = dg::construct<inner_vector>(src, std::forward<Params>(ps)...);
return t;
}
template<class To, class From, class Size, class ...Params>
To doConstruct( const From& src, RecursiveVectorTag, MPIVectorTag, Size size, Params&&... ps )
{
To t(size);
using inner_vector = typename To::value_type;
for (int i=0; i<(int)size; i++)
t[i] = dg::construct<inner_vector>(src, std::forward<Params>(ps)...);
return t;
}
template<class To, class From, class ...Params>
To doConstruct( const From& src, RecursiveVectorTag, RecursiveVectorTag, Params&&...ps )
{
unsigned size = src.size();
To t(size);
using inner_vector = typename To::value_type;
for (unsigned i=0; i<size; i++)
t[i] = dg::construct<inner_vector>(src[i], std::forward<Params>(ps)...);
return t;
}

template<class From, class To, class ...Params>
void doAssign( const From& src, To& to, SharedVectorTag, ArrayVectorTag, Params&&...ps )
{
for (unsigned i=0; i<to.size(); i++)
dg::assign(src, to[i], std::forward<Params>(ps)...);
}
template<class From, class To, class ...Params>
void doAssign( const From& src, To& to, MPIVectorTag, ArrayVectorTag, Params&&...ps )
{
for (unsigned i=0; i<to.size(); i++)
dg::assign(src, to[i], std::forward<Params>(ps)...);
}
template<class From, class To, class ...Params>
void doAssign( const From& src, To& to, RecursiveVectorTag, ArrayVectorTag, Params&&...ps )
{
for (unsigned i=0; i<to.size(); i++)
dg::assign(src[i], to[i], std::forward<Params>(ps)...);
}

template<class From, class To, class Size, class ...Params>
void doAssign( const From& src, To& to, SharedVectorTag, RecursiveVectorTag, Size size, Params&&... ps )
{
to.resize(size);
for (int i=0; i<(int)size; i++)
dg::assign(src, to[i], std::forward<Params>(ps)...);
}
template<class From, class To, class Size, class ...Params>
void doAssign( const From& src, To& to, MPIVectorTag, RecursiveVectorTag, Size size, Params&&... ps )
{
to.resize(size);
for (int i=0; i<(int)size; i++)
dg::assign(src, to[i], std::forward<Params>(ps)...);
}
template<class From, class To, class ...Params>
void doAssign( const From& src, To& to, RecursiveVectorTag, RecursiveVectorTag, Params&&...ps )
{
unsigned size = src.size();
to.resize(size);
for (unsigned i=0; i<size; i++)
dg::assign(src[i], to[i], std::forward<Params>(ps)...);
}

} 

namespace blas1
{


namespace detail
{


template< class Vector1, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector1& x1, const Vector2& x2, RecursiveVectorTag)
{
constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, Vector1, Vector1, Vector2>::value;
auto size = get_idx<vector_idx>(x1,x2).size();
std::vector<int64_t> acc( exblas::BIN_COUNT, (int64_t)0);
for( unsigned i=0; i<size; i++)
{
std::vector<int64_t> temp = doDot_superacc( do_get_vector_element(x1,i,get_tensor_category<Vector1>()), do_get_vector_element(x2,i,get_tensor_category<Vector2>()));
int imin = exblas::IMIN, imax = exblas::IMAX;
exblas::cpu::Normalize( &(temp[0]), imin, imax);
for( int k=exblas::IMIN; k<=exblas::IMAX; k++)
acc[k] += temp[k];
if( (i+1)%128 == 0)
{
imin = exblas::IMIN, imax = exblas::IMAX;
exblas::cpu::Normalize( &(acc[0]), imin, imax);
}
}
return acc;
}
#ifdef _OPENMP
template< class size_type, class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( RecursiveVectorTag, OmpTag, size_type size, Subroutine f, container&& x, Containers&&... xs)
{
if( !omp_in_parallel())
{
#pragma omp parallel
{
for( int i=0; i<(int)size; i++) {
dg::blas1::subroutine( f,
do_get_vector_element(std::forward<container>(x),i,get_tensor_category<container>()),
do_get_vector_element(std::forward<Containers>(xs),i,get_tensor_category<Containers>())...);
}
}
}
else 
for( int i=0; i<(int)size; i++) {
dg::blas1::subroutine( f,
do_get_vector_element(std::forward<container>(x),i,get_tensor_category<container>()),
do_get_vector_element(std::forward<Containers>(xs),i,get_tensor_category<Containers>())...);
}
}
#endif 



template<class size_type, class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( RecursiveVectorTag, AnyPolicyTag, size_type size, Subroutine f, container&& x, Containers&&... xs)
{
for( int i=0; i<(int)size; i++) {
dg::blas1::subroutine( f, do_get_vector_element(std::forward<container>(x),i,get_tensor_category<container>()), do_get_vector_element(std::forward<Containers>(xs),i,get_tensor_category<Containers>())...);
}
}

template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( RecursiveVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
constexpr unsigned vector_idx = find_if_v<dg::is_not_scalar, get_value_type<container>, container, Containers...>::value;
auto size = get_idx<vector_idx>( std::forward<container>(x), std::forward<Containers>(xs)...).size();
using vector_type = find_if_t<dg::has_not_any_policy, get_value_type<container>, container, Containers...>;
doSubroutine_dispatch( RecursiveVectorTag(), get_execution_policy<vector_type>(), size, f, std::forward<container>( x), std::forward<Containers>( xs)...);
}

template<class T, class ContainerType, class BinaryOp, class UnaryOp>
inline T doReduce( RecursiveVectorTag, const ContainerType& x, T init, BinaryOp
op, UnaryOp unary_op)
{
for ( unsigned u=0; u<x.size(); u++)
init = doReduce( get_tensor_category<decltype( x[u])>(), x[u], init, op,
unary_op);
return init;
}

} 
} 
} 

#endif 
