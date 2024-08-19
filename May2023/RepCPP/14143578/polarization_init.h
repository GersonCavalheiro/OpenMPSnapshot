#pragma once

#include "dg/algorithm.h"

namespace dg {
namespace mat {


template <class Geometry, class Matrix, class Container>
class PolChargeN
{
public:
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
PolChargeN(){}

PolChargeN( const Geometry& g,
direction dir = forward, value_type jfactor=1.):
PolChargeN( g, g.bcx(), g.bcy(), dir, jfactor)
{
}


PolChargeN( const Geometry& g, bc bcx, bc bcy,
direction dir = forward,
value_type jfactor=1.):
m_gamma(-0.5, {g, bcx, bcy, dir, jfactor})
{
m_ell.construct(g, bcx, bcy, dir, jfactor );
dg::assign(dg::evaluate(dg::zero,g), m_phi);
dg::assign(dg::evaluate(dg::one,g), m_temp);


m_tempx = m_tempx2 = m_tempy = m_tempy2 = m_temp;
dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);

dg::assign( dg::create::inv_volume(g),    m_inv_weights);
dg::assign( dg::create::volume(g),        m_weights);
dg::assign( dg::create::inv_weights(g),   m_precond);
m_temp = m_tempx = m_tempy = m_inv_weights;
m_chi=g.metric();
m_sigma = m_vol = dg::tensor::volume(m_chi);
dg::assign( dg::create::weights(g), m_weights_wo_vol);
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = PolChargeN( std::forward<Params>( ps)...);
}

template<class ContainerType0>
void set_phi( const ContainerType0& phi)
{
m_phi = phi;
}
template<class ContainerType0>
void set_dxphi( const ContainerType0& dxphi)
{
m_dxphi = dxphi;
}
template<class ContainerType0>
void set_dyphi( const ContainerType0& dyphi)
{
m_dyphi = dyphi;
}
template<class ContainerType0>
void set_lapphi( const ContainerType0& lapphi)
{
m_lapphi = lapphi;
}

const Container& weights()const {
return m_ell.weights();
}

const Container& precond()const {
return m_ell.precond();
}

template<class ContainerType0, class ContainerType1>
void operator()( const ContainerType0& x, ContainerType1& y){
symv( 1., x, 0., y);
}


template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y){
symv( 1., x, 0., y);
}

template<class ContainerType0, class ContainerType1>
void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
{
dg::blas1::copy(x, m_temp);
dg::blas1::plus( m_temp, -1.);
dg::blas2::gemv( m_rightx, m_temp, m_tempx2); 
dg::blas2::gemv( m_righty, m_temp, m_tempy2); 

dg::tensor::scalar_product2d(1., 1., m_dxphi, m_dyphi, m_chi, 1., m_tempx2, m_tempy2, 0., y); 
dg::blas1::pointwiseDot(m_lapphi, x, m_tempx);  

dg::blas1::axpbypgz(1.0, m_tempx, 1.0, m_temp, 1.0, y);
dg::blas1::scal(y,-1.0); 

}

private:
dg::Elliptic<Geometry,  Matrix, Container> m_ell;
dg::Helmholtz<Geometry,  Matrix, Container> m_gamma;
Container m_phi, m_dxphi,m_dyphi, m_lapphi, m_temp, m_tempx, m_tempx2, m_tempy, m_tempy2;

SparseTensor<Container> m_chi, m_metric;
Container m_sigma, m_vol;
Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;

Matrix m_rightx, m_righty;

};

}  

template< class G, class M, class V>
struct TensorTraits< mat::PolChargeN<G, M, V> >
{
using value_type      = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};

}  
