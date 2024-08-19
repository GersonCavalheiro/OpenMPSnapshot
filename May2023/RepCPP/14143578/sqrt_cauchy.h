#pragma once
#include <boost/math/special_functions.hpp>

#include "dg/algorithm.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



namespace dg {
namespace mat {


template<class Container>
struct SqrtCauchyInt
{
public:
using container_type = Container;
using value_type = dg::get_value_type<Container>;
SqrtCauchyInt() { }

SqrtCauchyInt( const Container& copyable)
{
m_helper = m_temp = m_helper3 = copyable;
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = SqrtCauchyInt( std::forward<Params>( ps)...);
}

const double& w() const{return m_w;}

template<class MatrixType>
auto make_denominator(MatrixType& A) const{
return [&A=A, &w = m_w] ( const auto& x, auto& y)
{
dg::blas2::symv(A, x, y); 
dg::blas1::axpby(w*w, x, 1., y); 
};
}


template<class MatrixType0, class MatrixType1, class ContainerType0,
class ContainerType1>
void operator()(MatrixType0&& A, MatrixType1&& wAinv, const ContainerType0&
b, ContainerType1& x, std::array<value_type,2> EVs, unsigned
steps,  int exp = +1)
{
dg::blas1::copy(0., m_helper3);
value_type s=0.;
value_type c=0.;
value_type d=0.;
m_w=0.;
value_type t=0.;
value_type minEV = EVs[0], maxEV = EVs[1];
value_type sqrtminEV = std::sqrt(minEV);
const value_type k2 = minEV/maxEV;
const value_type sqrt1mk2 = std::sqrt(1.-k2);
const value_type Ks=boost::math::ellint_1(sqrt1mk2 );
const value_type fac = 2.* Ks*sqrtminEV/(M_PI*steps);
for (unsigned j=1; j<steps+1; j++)
{
t  = (j-0.5)*Ks/steps; 
c = 1./boost::math::jacobi_cn(sqrt1mk2, t);
s = boost::math::jacobi_sn(sqrt1mk2, t)*c;
d = boost::math::jacobi_dn(sqrt1mk2, t)*c;
m_w = sqrtminEV*s;
dg::blas1::axpby(c*d, b, 0.0 , m_helper); 
dg::blas2::symv( std::forward<MatrixType1>(wAinv), m_helper, m_temp);

dg::blas1::axpby(fac, m_temp, 1.0, m_helper3); 
}
if( exp > 0)
dg::blas2::symv(A, m_helper3, x);
else
dg::blas1::copy( m_helper3, x);
}

private:
Container m_helper, m_temp, m_helper3;
value_type m_w;
};


template< class Container>
struct DirectSqrtCauchy
{
public:
using container_type = Container;
using value_type = dg::get_value_type<Container>;
DirectSqrtCauchy() {}

template<class MatrixType>
DirectSqrtCauchy(
MatrixType& A,
const Container& weights,
value_type epsCG,
unsigned iterCauchy,
std::array<value_type,2> EVs, int exp)
{
m_pcg.construct( weights, 10000);
Container m_temp = weights;
m_iterCauchy = iterCauchy;
m_cauchysqrtint.construct(weights);
m_EVs = EVs;
m_A = [&]( const Container& x, Container& y){
dg::blas2::symv( A, x, y);
};
m_op = m_cauchysqrtint.make_denominator(A);
m_wAinv = [&, eps = epsCG, w = weights]
( const Container& x, Container& y){
m_pcg.solve( m_op, y, x, 1., w, eps);
};
m_exp = exp;
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = DirectSqrtCauchy( std::forward<Params>( ps)...);
}

unsigned operator()(const Container& b, Container& x)
{
m_cauchysqrtint(m_A, m_wAinv, b, x, m_EVs, m_iterCauchy, m_exp);
return m_iterCauchy;
}
private:
unsigned m_iterCauchy;
std::function<void ( const Container&, Container&)> m_A, m_wAinv, m_op;
dg::PCG<Container> m_pcg;
dg::mat::SqrtCauchyInt<Container> m_cauchysqrtint;
std::array<value_type,2> m_EVs;
int m_exp;
};


} 
} 
