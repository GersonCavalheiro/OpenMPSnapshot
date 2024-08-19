#pragma once

#include "mpi_vector.h"
#include "memory.h"
#include "timer.h"


namespace dg {
namespace blas2 {
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( MatrixType&& M,
const ContainerType1& x,
ContainerType2& y);
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
void stencil( FunctorType f, MatrixType&& M,
const ContainerType1& x,
ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( get_value_type<ContainerType1> alpha,
MatrixType&& M,
const ContainerType1& x,
get_value_type<ContainerType1> beta,
ContainerType2& y);
}



template<class LocalMatrixInner, class LocalMatrixOuter, class Collective >
struct RowColDistMat
{
using value_type = get_value_type<LocalMatrixInner>;
RowColDistMat(){}


RowColDistMat( const LocalMatrixInner& inside, const LocalMatrixOuter& outside, const Collective& c):
m_i(inside), m_o(outside), m_c(c), m_buffer( c.allocate_buffer()) { }


template< class OtherMatrixInner, class OtherMatrixOuter, class OtherCollective>
RowColDistMat( const RowColDistMat<OtherMatrixInner, OtherMatrixOuter, OtherCollective>& src):
m_i(src.inner_matrix()), m_o( src.outer_matrix()), m_c(src.collective()), m_buffer( m_c.allocate_buffer()) { }
const LocalMatrixInner& inner_matrix() const{return m_i;}
LocalMatrixInner& inner_matrix(){return m_i;}
const LocalMatrixOuter& outer_matrix() const{return m_o;}
LocalMatrixOuter& outer_matrix(){return m_o;}
const Collective& collective() const{return m_c;}


template<class ContainerType1, class ContainerType2>
void symv( value_type alpha, const ContainerType1& x, value_type beta, ContainerType2& y) const
{
if( !m_c.isCommunicating()) 
{
dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
return;

}
int result;
MPI_Comm_compare( x.communicator(), y.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm_compare( x.communicator(), m_c.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);

MPI_Request rqst[4];
const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
value_type * y_ptr = thrust::raw_pointer_cast(y.data().data());
m_c.global_gather_init( x_ptr, m_buffer.data(), rqst);
dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
m_c.global_gather_wait( x_ptr, m_buffer.data(), rqst);
const value_type** b_ptr = thrust::raw_pointer_cast(m_buffer.data().data());
m_o.symv( SharedVectorTag(), get_execution_policy<ContainerType1>(), alpha, b_ptr, 1., y_ptr);
}


template<class ContainerType1, class ContainerType2>
void symv( const ContainerType1& x, ContainerType2& y) const
{
if( !m_c.isCommunicating()) 
{
dg::blas2::symv( m_i, x.data(), y.data());
return;

}
int result;
MPI_Comm_compare( x.communicator(), y.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm_compare( x.communicator(), m_c.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);

MPI_Request rqst[4];
const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
value_type * y_ptr = thrust::raw_pointer_cast(y.data().data());
m_c.global_gather_init( x_ptr, m_buffer.data(), rqst);
dg::blas2::symv( m_i, x.data(), y.data());
m_c.global_gather_wait( x_ptr, m_buffer.data(), rqst);
const value_type** b_ptr = thrust::raw_pointer_cast(m_buffer.data().data());
m_o.symv( SharedVectorTag(), get_execution_policy<ContainerType1>(), 1., b_ptr, 1., y_ptr);
}

private:
LocalMatrixInner m_i;
LocalMatrixOuter m_o;
Collective m_c;
Buffer< typename Collective::buffer_type>  m_buffer;
};


enum dist_type
{
row_dist=0, 
col_dist=1 
};


template<class LocalMatrix, class Collective >
struct MPIDistMat
{
using value_type = get_value_type<LocalMatrix>;
MPIDistMat( ) { }

MPIDistMat( const LocalMatrix& m, const Collective& c, enum dist_type dist = row_dist):
m_m(m), m_c(c), m_buffer( c.allocate_buffer()), m_dist( dist) { }


template< class OtherMatrix, class OtherCollective>
MPIDistMat( const MPIDistMat<OtherMatrix, OtherCollective>& src):
m_m(src.matrix()), m_c(src.collective()), m_buffer( m_c->allocate_buffer()), m_dist(src.get_dist()) { }

const LocalMatrix& matrix() const{return m_m;}

const Collective& collective() const{return *m_c;}

enum dist_type get_dist() const {return m_dist;}
void set_dist(enum dist_type dist){m_dist=dist;}

template<class ContainerType1, class ContainerType2>
void symv( value_type alpha, const ContainerType1& x, value_type beta, ContainerType2& y) const
{
if( !m_c->isCommunicating()) 
{
dg::blas2::symv( alpha, m_m, x.data(), beta, y.data());
return;

}
int result;
MPI_Comm_compare( x.communicator(), y.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm_compare( x.communicator(), m_c->communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
if( m_dist == row_dist){
const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
m_c->global_gather( x_ptr, m_buffer.data());
dg::blas2::symv( alpha, m_m, m_buffer.data(), beta, y.data());
}
if( m_dist == col_dist){
dg::blas2::symv( alpha, m_m, x.data(), beta, m_buffer.data());
value_type * y_ptr = thrust::raw_pointer_cast(y.data().data());
m_c->global_scatter_reduce( m_buffer.data(), y_ptr);
}
}
template<class ContainerType1, class ContainerType2>
void symv( const ContainerType1& x, ContainerType2& y) const
{
if( !m_c->isCommunicating()) 
{
dg::blas2::symv( m_m, x.data(), y.data());
return;

}
int result;
MPI_Comm_compare( x.communicator(), y.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm_compare( x.communicator(), m_c->communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
if( m_dist == row_dist){
const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
m_c->global_gather( x_ptr, m_buffer.data());
dg::blas2::symv( m_m, m_buffer.data(), y.data());
}
if( m_dist == col_dist){
dg::blas2::symv( m_m, x.data(), m_buffer.data());
value_type * y_ptr = thrust::raw_pointer_cast(y.data().data());
m_c->global_scatter_reduce( m_buffer.data(), y_ptr);
}
}
template<class Functor, class ContainerType1, class ContainerType2>
void stencil( const Functor f, const ContainerType1& x, ContainerType2& y) const
{
if( !m_c->isCommunicating()) 
{
dg::blas2::stencil( f, m_m, x.data(), y.data());
return;

}
int result;
MPI_Comm_compare( x.communicator(), y.communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
MPI_Comm_compare( x.communicator(), m_c->communicator(), &result);
assert( result == MPI_CONGRUENT || result == MPI_IDENT);
if( m_dist == row_dist){
const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
m_c->global_gather( x_ptr, m_buffer.data());
dg::blas2::stencil( f, m_m, m_buffer.data(), y.data());
}
if( m_dist == col_dist){
throw Error( Message(_ping_)<<"stencil cannot be used with a column distributed mpi matrix!");
}
}

private:
LocalMatrix m_m;
ClonePtr<Collective> m_c;
Buffer< typename Collective::container_type> m_buffer;
enum dist_type m_dist;
};

template<class LI, class LO, class C>
struct TensorTraits<RowColDistMat<LI,LO, C> >
{
using value_type = get_value_type<LI>;
using tensor_category = MPIMatrixTag;
};

template<class L, class C>
struct TensorTraits<MPIDistMat<L, C> >
{
using value_type = get_value_type<L>;
using tensor_category = MPIMatrixTag;
};

} 
