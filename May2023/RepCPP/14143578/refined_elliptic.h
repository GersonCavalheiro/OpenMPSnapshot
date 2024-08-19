#pragma once

#include "topology/interpolation.h"
#include "topology/projection.h"
#include "elliptic.h"
#include "topology/refined_grid.h"
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif


namespace dg
{


template < class Geometry,class IMatrix, class Matrix, class Container>
class RefinedElliptic
{
public:
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;

RefinedElliptic( const Geometry& g_coarse, const Geometry& g_fine, direction dir = forward): RefinedElliptic( g_coarse, g_fine, g_fine.bcx(), g_fine.bcy(), dir){}


RefinedElliptic( const Geometry& g_coarse, const Geometry& g_fine, bc bcx, bc bcy, direction dir = forward):
elliptic_( g_fine, bcx, bcy, dir)
{
construct( g_coarse, g_fine, bcx, bcy, dir);
}


template<class ContainerType0>
void set_chi( const ContainerType0& chi)
{
elliptic_.set_chi( chi);
}

const Container& weights()const {return weights_;}

const Container& precond()const {return precond_;}


template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y)
{
dg::blas2::gemv( Q_, x, temp1_);
elliptic_.symv( temp1_, temp2_);
dg::blas2::gemv( P_, temp2_, y);
return;
}

private:
void construct( const Geometry& g_coarse, const Geometry& g_fine, bc bcx, bc bcy, direction dir)
{
dg::blas2::transfer( dg::create::interpolation( g_fine, g_coarse), Q_);
dg::blas2::transfer( dg::create::projection( g_coarse, g_fine), P_);

dg::assign( dg::evaluate( dg::one, g_fine), temp1_);
dg::assign( dg::evaluate( dg::one, g_fine), temp2_);
dg::assign( dg::create::weights( g_coarse), weights_);
dg::assign( dg::evaluate( dg::one ,g_coarse), precond_);
vol_ = dg::tensor::volume( g_fine.metric());

}
IMatrix P_, Q_;
Elliptic<Geometry, Matrix, Container> elliptic_;
Container temp1_, temp2_;
Container weights_, precond_;
Container vol_;
};


template< class G, class IM, class M, class V>
struct TensorTraits< RefinedElliptic<G, IM, M, V> >
{
using value_type  = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};


} 

