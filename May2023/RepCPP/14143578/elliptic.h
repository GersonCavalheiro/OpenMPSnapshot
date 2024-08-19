#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/memory.h"
#include "topology/evaluation.h"
#include "topology/derivatives.h"
#ifdef MPI_VERSION
#include "topology/mpi_derivatives.h"
#include "topology/mpi_evaluation.h"
#endif
#include "topology/geometry.h"


namespace dg
{




template <class Geometry, class Matrix, class Container>
class Elliptic1d
{
public:
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
Elliptic1d() = default;

Elliptic1d( const Geometry& g,
direction dir = forward, value_type jfactor=1.):
Elliptic1d( g, g.bcx(), dir, jfactor)
{
}


Elliptic1d( const Geometry& g, bc bcx,
direction dir = forward,
value_type jfactor=1.)
{
m_jfactor=jfactor;
dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
dg::blas2::transfer( dg::create::jump( g, bcx),   m_jumpX);

dg::assign( dg::create::weights(g),       m_weights);
dg::assign( dg::evaluate( dg::one, g),    m_precond);
m_tempx = m_sigma = m_precond;
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = Elliptic1d( std::forward<Params>( ps)...);
}


template<class ContainerType0>
void set_chi( const ContainerType0& sigma)
{
dg::blas1::copy( sigma, m_sigma);
dg::blas1::pointwiseDivide( 1., sigma, m_precond);
}


const Container& weights()const {
return m_weights;
}

const Container& precond()const {
return m_precond;
}
void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
value_type get_jfactor() const {return m_jfactor;}
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
dg::blas2::gemv( m_rightx, x, m_tempx);
dg::blas1::pointwiseDot( m_tempx, m_sigma, m_tempx);
dg::blas2::symv( -alpha, m_leftx, m_tempx, beta, y);
if( 0.0 != m_jfactor )
{
dg::blas2::symv( m_jfactor*alpha, m_jumpX, x, 1., y);
}
}

private:
Matrix m_leftx, m_rightx, m_jumpX;
Container m_weights, m_precond;
Container m_tempx;
Container m_sigma;
value_type m_jfactor;
};


template <class Geometry, class Matrix, class Container>
class Elliptic2d
{
public:
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
Elliptic2d() = default;

Elliptic2d( const Geometry& g,
direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
Elliptic2d( g, g.bcx(), g.bcy(), dir, jfactor, chi_weight_jump)
{
}


Elliptic2d( const Geometry& g, bc bcx, bc bcy,
direction dir = forward,
value_type jfactor=1., bool chi_weight_jump = false)
{
m_jfactor=jfactor;
m_chi_weight_jump = chi_weight_jump;
dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);

dg::assign( dg::create::volume(g),        m_weights);
dg::assign( dg::evaluate( dg::one, g),    m_precond);
m_temp = m_tempx = m_tempy = m_weights;
m_chi=g.metric();
m_sigma = m_vol = dg::tensor::volume(m_chi);
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = Elliptic2d( std::forward<Params>( ps)...);
}


template<class ContainerType0>
void set_chi( const ContainerType0& sigma)
{
dg::blas1::pointwiseDot( sigma, m_vol, m_sigma);
dg::blas1::pointwiseDivide( 1., sigma, m_precond);
}

template<class ContainerType0>
void set_chi( const SparseTensor<ContainerType0>& tau)
{
m_chi = SparseTensor<Container>(tau);
}


const Container& weights()const {
return m_weights;
}

const Container& precond()const {
return m_precond;
}

void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}

value_type get_jfactor() const {return m_jfactor;}

void set_jump_weighting( bool jump_weighting) {m_chi_weight_jump = jump_weighting;}

bool get_jump_weighting() const {return m_chi_weight_jump;}

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
dg::blas2::gemv( m_rightx, x, m_tempx); 
dg::blas2::gemv( m_righty, x, m_tempy); 

dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);

dg::blas2::symv( m_lefty, m_tempy, m_temp);
dg::blas2::symv( -1., m_leftx, m_tempx, -1., m_temp);

if( 0.0 != m_jfactor )
{
if(m_chi_weight_jump)
{
dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);
dg::blas1::axpbypgz(1.0,m_tempx,1.0,m_tempy,1.0,m_temp);
}
else
{
dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
}
}
dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
}


template<class ContainerType0, class ContainerType1>
void variation(const ContainerType0& phi, ContainerType1& sigma){
variation(1., 1., phi, 0., sigma);
}

template<class ContainerTypeL, class ContainerType0, class ContainerType1>
void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
variation(1.,lambda, phi, 0., sigma);
}

template<class ContainerTypeL, class ContainerType0, class ContainerType1>
void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
{
dg::blas2::gemv( m_rightx, phi, m_tempx); 
dg::blas2::gemv( m_righty, phi, m_tempy); 
dg::tensor::scalar_product2d(alpha, lambda, m_tempx, m_tempy, m_chi, lambda, m_tempx, m_tempy, beta, sigma);
}


private:
Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY;
Container m_weights, m_precond;
Container m_tempx, m_tempy, m_temp;
SparseTensor<Container> m_chi;
Container m_sigma, m_vol;
value_type m_jfactor;
bool m_chi_weight_jump;
};

template <class Geometry, class Matrix, class Container>
using Elliptic = Elliptic2d<Geometry, Matrix, Container>;


template <class Geometry, class Matrix, class Container>
class Elliptic3d
{
public:
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
Elliptic3d() = default;

Elliptic3d( const Geometry& g, direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
Elliptic3d( g, g.bcx(), g.bcy(), g.bcz(), dir, jfactor, chi_weight_jump)
{
}


Elliptic3d( const Geometry& g, bc bcx, bc bcy, bc bcz, direction dir = forward, value_type jfactor = 1., bool chi_weight_jump = false)
{
m_jfactor=jfactor;
m_chi_weight_jump = chi_weight_jump;
dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);
if( g.nz() == 1)
{
dg::blas2::transfer( dg::create::dz( g, bcz, dg::centered), m_rightz);
dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dg::centered)), m_leftz);
m_addJumpZ = false;
}
else
{
dg::blas2::transfer( dg::create::dz( g, bcz, dir), m_rightz);
dg::blas2::transfer( dg::create::dz( g, inverse( bcz), inverse(dir)), m_leftz);
dg::blas2::transfer( dg::create::jumpZ( g, bcz),   m_jumpZ);
m_addJumpZ = true;
}

dg::assign( dg::create::volume(g),        m_weights);
dg::assign( dg::evaluate( dg::one, g),    m_precond);
m_temp = m_tempx = m_tempy = m_tempz = m_weights;
m_chi=g.metric();
m_sigma = m_vol = dg::tensor::volume(m_chi);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = Elliptic3d( std::forward<Params>( ps)...);
}

template<class ContainerType0>
void set_chi( const ContainerType0& sigma)
{
dg::blas1::pointwiseDot( sigma, m_vol, m_sigma);
dg::blas1::pointwiseDivide( 1., sigma, m_precond);
}

template<class ContainerType0>
void set_chi( const SparseTensor<ContainerType0>& tau)
{
m_chi = SparseTensor<Container>(tau);
}

const Container& weights()const {
return m_weights;
}
const Container& precond()const {
return m_precond;
}
void set_jfactor( value_type new_jfactor) {m_jfactor = new_jfactor;}
value_type get_jfactor() const {return m_jfactor;}
void set_jump_weighting( bool jump_weighting) {m_chi_weight_jump = jump_weighting;}
bool get_jump_weighting() const {return m_chi_weight_jump;}


void set_compute_in_2d( bool compute_in_2d ) {
m_multiplyZ = !compute_in_2d;
}

template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y){
symv( 1, x, 0, y);
}
template<class ContainerType0, class ContainerType1>
void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
{
dg::blas2::gemv( m_rightx, x, m_tempx); 
dg::blas2::gemv( m_righty, x, m_tempy); 
if( m_multiplyZ )
{
dg::blas2::gemv( m_rightz, x, m_tempz); 

dg::tensor::multiply3d(m_sigma, m_chi, m_tempx, m_tempy, m_tempz, 0., m_tempx, m_tempy, m_tempz);
dg::blas2::symv( -1., m_leftz, m_tempz, 0., m_temp);
dg::blas2::symv( -1., m_lefty, m_tempy, 1., m_temp);
}
else
{
dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy, 0., m_tempx, m_tempy);
dg::blas2::symv( -1.,m_lefty, m_tempy, 0., m_temp);
}
dg::blas2::symv( -1., m_leftx, m_tempx, 1., m_temp);

if( 0 != m_jfactor )
{
if(m_chi_weight_jump)
{
dg::blas2::symv( m_jfactor, m_jumpX, x, 0., m_tempx);
dg::blas2::symv( m_jfactor, m_jumpY, x, 0., m_tempy);
if( m_addJumpZ)
{
dg::blas2::symv( m_jfactor, m_jumpZ, x, 0., m_tempz);
dg::tensor::multiply3d(m_sigma, m_chi, m_tempx, m_tempy,
m_tempz, 0., m_tempx, m_tempy, m_tempz);
}
else
dg::tensor::multiply2d(m_sigma, m_chi, m_tempx, m_tempy,
0., m_tempx, m_tempy);

dg::blas1::axpbypgz(1., m_tempx, 1., m_tempy, 1., m_temp);
if( m_addJumpZ)
dg::blas1::axpby( 1., m_tempz, 1., m_temp);
}
else
{
dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_temp);
dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_temp);
if( m_addJumpZ)
dg::blas2::symv( m_jfactor, m_jumpZ, x, 1., m_temp);
}
}
dg::blas1::pointwiseDivide( alpha, m_temp, m_vol, beta, y);
}

template<class ContainerType0, class ContainerType1>
void variation(const ContainerType0& phi, ContainerType1& sigma){
variation(1.,1., phi, 0., sigma);
}
template<class ContainerTypeL, class ContainerType0, class ContainerType1>
void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
variation(1.,lambda, phi, 0., sigma);
}
template<class ContainerTypeL, class ContainerType0, class ContainerType1>
void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
{
dg::blas2::gemv( m_rightx, phi, m_tempx); 
dg::blas2::gemv( m_righty, phi, m_tempy); 
if( m_multiplyZ)
dg::blas2::gemv( m_rightz, phi, m_tempz); 
else
dg::blas1::scal( m_tempz, 0.);
dg::tensor::scalar_product3d(alpha, lambda,  m_tempx, m_tempy, m_tempz, m_chi, lambda, m_tempx, m_tempy, m_tempz, beta, sigma);
}

private:
Matrix m_leftx, m_lefty, m_leftz, m_rightx, m_righty, m_rightz, m_jumpX, m_jumpY, m_jumpZ;
Container m_weights, m_precond;
Container m_tempx, m_tempy, m_tempz, m_temp;
SparseTensor<Container> m_chi;
Container m_sigma, m_vol;
value_type m_jfactor;
bool m_multiplyZ = true, m_addJumpZ = false;
bool m_chi_weight_jump;
};
template< class G, class M, class V>
struct TensorTraits< Elliptic1d<G, M, V> >
{
using value_type      = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};
template< class G, class M, class V>
struct TensorTraits< Elliptic2d<G, M, V> >
{
using value_type      = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};

template< class G, class M, class V>
struct TensorTraits< Elliptic3d<G, M, V> >
{
using value_type      = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};

} 
