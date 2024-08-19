#pragma once

#include <cassert>

#include "dg/algorithm.h"


namespace dg{
namespace mat{


template< class Geometry, class Matrix, class Container>
struct TensorElliptic
{
using container_type = Container;
using geometry_type = Geometry;
using matrix_type = Matrix;
using value_type = get_value_type<Container>;
TensorElliptic() {}

TensorElliptic( const Geometry& g, direction dir = dg::centered, value_type jfactor=1.):
TensorElliptic( g, g.bcx(), g.bcy(), dir, jfactor)
{
}

TensorElliptic( const Geometry& g, bc bcx, bc bcy, direction dir = dg::centered, value_type jfactor=1.)
{
m_jfactor=jfactor;
m_laplaceM_chi.construct( g, bcx, bcy, dir, jfactor);
m_laplaceM_iota.construct( g, bcx, bcy, dir, jfactor);
dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

dg::assign( dg::evaluate( dg::one, g), m_temp);
m_tempx = m_tempy = m_tempxy = m_tempyx = m_iota  = m_helper = m_temp;

m_chi=g.metric();
m_metric=g.metric();
m_vol=dg::tensor::volume(m_chi); 
dg::tensor::scal( m_chi, m_vol); 
dg::assign( dg::evaluate(dg::one, g), m_sigma);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = TensorElliptic( std::forward<Params>( ps)...);
}

const Container& weights()const {return m_laplaceM_chi.weights();}

const Container& precond()const {return m_laplaceM_chi.precond();}

template<class ContainerType0>
void set_chi( const ContainerType0& chi) {m_laplaceM_chi.set_chi(chi); }

template<class ContainerType0>
void set_iota( const ContainerType0& iota) {m_iota=iota; }

void variation(const Container& phi, const value_type& alpha, const Container& chi, Container& varphi)
{
dg::blas2::symv( m_rightx, phi, m_tempx); 
dg::blas2::symv(-1.0, m_leftx, m_tempx, 0.0, m_helper); 
dg::blas2::symv(-1.0, m_righty, m_tempx, 0.0, m_tempyx); 
dg::blas2::symv( m_righty, phi, m_tempy); 
dg::blas2::symv(-1.0, m_lefty, m_tempy, 0.0, m_temp); 
dg::blas2::symv(-1.0, m_rightx, m_tempy, 0.0, m_tempxy); 

dg::blas2::symv( m_jfactor, m_jumpX, phi, 1., m_helper);
dg::blas2::symv( m_jfactor, m_jumpY, phi, 1., m_temp);

dg::blas1::pointwiseDot(alpha, m_temp,     m_temp,  alpha, m_helper,  m_helper,  0., varphi);
dg::blas1::pointwiseDot(alpha, m_tempxy, m_tempxy,  alpha, m_tempyx,  m_tempyx,  1., varphi);
dg::blas1::pointwiseDot(varphi, chi, varphi);

dg::blas2::symv(m_laplaceM_iota, phi, m_temp);
dg::blas1::pointwiseDot(alpha/2., chi, m_temp, m_temp, -1., varphi); 

tensor::multiply2d( m_metric, m_tempx, m_tempy, m_temp, m_helper);
dg::blas1::pointwiseDot( 1., m_temp, m_tempx, 1., m_helper, m_tempy, 0., m_temp); 
dg::blas1::axpby(-0.5, m_temp, -0.5, varphi);
dg::blas1::pointwiseDot(chi, varphi, varphi);

}

template<class ContainerType0, class ContainerType1>
void operator()( const ContainerType0& x, ContainerType1& y){
symv( 1, x, 0, y);
}

template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y){
symv( 1, x, 0, y);
}

template<class ContainerType0, class ContainerType1>
void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
{
dg::blas2::symv( m_rightx, x, m_helper); 
dg::blas2::symv(-1.0, m_leftx, m_helper, 0.0, m_tempx); 
dg::blas2::symv(-1.0, m_righty, m_helper, 0.0, m_tempyx); 
dg::blas2::symv( m_righty, x, m_helper); 
dg::blas2::symv(-1.0, m_lefty, m_helper, 0.0, m_tempy); 
dg::blas2::symv(-1.0, m_rightx, m_helper, 0.0, m_tempxy); 

dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_tempx);
dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_tempy);

dg::blas1::pointwiseDot( 1., m_tempx,  m_iota,  m_vol, 0., m_tempx);
dg::blas1::pointwiseDot( 1., m_tempyx, m_iota, m_vol, 0., m_tempyx);
dg::blas1::pointwiseDot( 1., m_tempy,  m_iota,  m_vol, 0., m_tempy);
dg::blas1::pointwiseDot( 1., m_tempxy, m_iota, m_vol, 0., m_tempxy);

dg::blas2::symv( m_rightx, m_tempx, m_helper);
dg::blas2::symv(-1.0, m_leftx, m_helper, 0.0, m_temp);  
dg::blas2::symv( m_leftx, m_tempyx, m_helper);
dg::blas2::symv(-1.0, m_lefty, m_helper, 1.0, m_temp); 
dg::blas2::symv( m_righty, m_tempy, m_helper);
dg::blas2::symv(-1.0, m_lefty, m_helper, 1.0, m_temp); 
dg::blas2::symv( m_lefty, m_tempxy, m_helper);
dg::blas2::symv(-1.0, m_leftx, m_helper, 1.0, m_temp);   

dg::blas2::symv( m_jfactor, m_jumpX, m_tempx, 1., m_temp);
dg::blas2::symv( m_jfactor, m_jumpY, m_tempy, 1., m_temp);

dg::blas1::pointwiseDivide(m_temp, m_vol, m_temp); 

dg::blas2::symv( m_laplaceM_iota, x, m_tempx);
dg::blas1::pointwiseDot( m_iota, m_tempx, m_tempx);
dg::blas2::symv(-1.0, m_laplaceM_iota, m_tempx, 2.0, m_temp);

dg::blas2::symv(1.0, m_laplaceM_chi, x, 1., m_temp);

dg::blas1::axpby(alpha, m_temp, beta, y);

}

private:
bc inverse( bc bound)
{
if( bound == DIR) return NEU;
if( bound == NEU) return DIR;
if( bound == DIR_NEU) return NEU_DIR;
if( bound == NEU_DIR) return DIR_NEU;
return PER;
}
direction inverse( direction dir)
{
if( dir == forward) return backward;
if( dir == backward) return forward;
return centered;
}
Elliptic<Geometry, Matrix, Container> m_laplaceM_chi, m_laplaceM_iota;
Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY;
Container m_temp, m_tempx, m_tempy, m_tempxy, m_tempyx, m_iota, m_helper;
SparseTensor<Container> m_chi;
SparseTensor<Container> m_metric;
Container m_sigma, m_vol;
value_type m_jfactor;
};


} 
template< class G, class M, class V>
struct TensorTraits< mat::TensorElliptic<G, M, V> >
{
using value_type  = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};
} 

