#pragma once

#include <cassert>

#include "blas.h"
#include "elliptic.h"


namespace dg{


template<class Matrix, class Container>
struct GeneralHelmholtz
{
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;

GeneralHelmholtz() = default;


GeneralHelmholtz( value_type alpha, Matrix matrix):
m_alpha(alpha), m_matrix(matrix), m_chi( m_matrix.weights())
{
dg::blas1::copy( 1., m_chi);
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = GeneralHelmholtz( std::forward<Params>( ps)...);
}

const Container& weights()const {return m_matrix.weights();}
const Container& precond()const {return m_matrix.precond();}


template<class ContainerType0, class ContainerType1>
void symv( const ContainerType0& x, ContainerType1& y)
{
if( m_alpha != 0)
blas2::symv( m_matrix, x, y);
dg::blas1::pointwiseDot( 1., m_chi, x, -m_alpha, y);

}

Matrix& matrix(){
return m_matrix;
}
const Matrix& matrix()const{
return m_matrix;
}

value_type& alpha( ){  return m_alpha;}

value_type alpha( ) const  {return m_alpha;}

template<class ContainerType0>
void set_chi( const ContainerType0& chi) {
dg::blas1::copy( chi, m_chi);
}

const Container& chi() const{return m_chi;}
private:
value_type m_alpha;
Matrix m_matrix;
Container m_chi;
};

template<class Geometry, class Matrix, class Container>
using Helmholtz = GeneralHelmholtz<dg::Elliptic2d<Geometry,Matrix,Container>, Container>;
template<class Geometry, class Matrix, class Container>
using Helmholtz1d = GeneralHelmholtz<dg::Elliptic1d<Geometry,Matrix,Container>, Container>;
template<class Geometry, class Matrix, class Container>
using Helmholtz2d = GeneralHelmholtz<dg::Elliptic2d<Geometry,Matrix,Container>, Container>;
template<class Geometry, class Matrix, class Container>
using Helmholtz3d = GeneralHelmholtz<dg::Elliptic3d<Geometry,Matrix,Container>, Container>;


template< class Geometry, class Matrix, class Container>
struct Helmholtz2
{
using container_type = Container;
using geometry_type = Geometry;
using matrix_type = Matrix;
using value_type = get_value_type<Container>;
Helmholtz2() {}

Helmholtz2( const Geometry& g, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.)
{
construct( g, alpha, dir, jfactor);
}

Helmholtz2( const Geometry& g, bc bcx, bc bcy, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.)
{
construct( g, bcx, bcy, alpha, dir, jfactor);
}
void construct( const Geometry& g, bc bcx, bc bcy, value_type alpha = 1, direction dir = dg::forward, value_type jfactor = 1.)
{
m_laplaceM.construct( g, bcx, bcy, dir, jfactor);
dg::assign( dg::evaluate( dg::one, g), temp1_);
dg::assign( dg::evaluate( dg::one, g), temp2_);
alpha_ = alpha;
}
void construct( const Geometry& g, value_type alpha = 1, direction dir = dg::forward, value_type jfactor = 1.)
{
construct( g, g.bcx(), g.bcy(), alpha, dir, jfactor);
}

void symv(const Container& x, Container& y)
{
if( alpha_ != 0)
{
blas2::symv( m_laplaceM, x, temp1_); 
blas1::pointwiseDivide(temp1_, chi_, y); 
blas2::symv( m_laplaceM, y, temp2_);
}
blas1::pointwiseDot( chi_, x, y); 
blas1::axpby( 1., y, -2.*alpha_, temp1_, y);
blas1::axpby( alpha_*alpha_, temp2_, 1., y, y);
}
const Container& weights()const {return m_laplaceM.weights();}

const Container& precond()const {return m_laplaceM.precond();}

value_type& alpha( ){  return alpha_;}

value_type alpha( ) const  {return alpha_;}

void set_chi( const Container& chi) {chi_=chi; }

const Container& chi()const {return chi_;}
private:
Elliptic2d<Geometry, Matrix, Container> m_laplaceM;
Container temp1_, temp2_;
Container chi_;
value_type alpha_;
};
template< class M, class V>
struct TensorTraits< GeneralHelmholtz<M, V> >
{
using value_type  = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};
template< class G, class M, class V>
struct TensorTraits< Helmholtz2<G, M, V> >
{
using value_type  = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};


} 

