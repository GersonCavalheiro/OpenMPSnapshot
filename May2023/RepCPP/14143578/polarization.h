#pragma once
#include "dg/algorithm.h"

#include "lanczos.h"
#include "matrixsqrt.h"
#include "matrixfunction.h"
#include "tensorelliptic.h"

namespace dg {
namespace mat {


template <class Geometry, class Matrix, class Container>
class PolCharge
{
public:
using value_type = get_value_type<Container>;
PolCharge(){}

PolCharge(value_type alpha, std::vector<value_type> eps_gamma,
const Geometry& g, direction dir = forward, value_type jfactor=1.,
std::string mode = "df", bool commute = false):
PolCharge( alpha, eps_gamma, g, g.bcx(), g.bcy(), dir, jfactor, mode,
commute)
{ }

PolCharge( value_type alpha, std::vector<value_type> eps_gamma,
const Geometry& g, bc bcx, bc bcy, direction dir = forward,
value_type jfactor=1., std::string mode = "df", bool commute = false)
{
m_alpha = alpha;
m_eps_gamma = eps_gamma;
m_mode = mode;
m_commute = commute;
m_temp2 = dg::evaluate(dg::zero, g);
m_temp =  m_temp2;
m_temp2_ex.set_max(1, m_temp2);
m_temp_ex.set_max(1, m_temp);
if (m_mode == "df")
{
m_ell.construct(g, bcx, bcy, dir, jfactor );
m_multi_g.construct(g, 3);
for( unsigned u=0; u<3; u++)
{
m_multi_gamma.push_back( {m_alpha, {m_multi_g.grid(u), bcx, bcy,
dir, jfactor}});
}
}
if (m_mode == "ff")
{
m_ell.construct(g, bcx, bcy, dir, jfactor );
m_multi_gamma.resize(1);
m_multi_gamma.resize(1);
m_multi_gamma[0].construct( m_alpha, dg::Elliptic<Geometry,
Matrix, Container>{g, bcx, bcy, dir, jfactor});

m_inv_sqrt.construct( m_multi_gamma[0], -1,
m_multi_gamma[0].weights(), m_eps_gamma[0]);
}
if (m_mode == "ffO4")
{
m_tensorell.construct(g, bcx, bcy, dir, jfactor);
m_multi_g.construct(g, 3);
for( unsigned u=0; u<3; u++)
{
m_multi_gamma.push_back({ m_alpha, {m_multi_g.grid(u), bcx, bcy,
dir, jfactor}});
}
}
}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = PolCharge( std::forward<Params>( ps)...);
}

template<class ContainerType0>
void set_chi( const ContainerType0& sigma)
{
if (m_mode == "ff")
m_ell.set_chi(sigma);
if (m_mode == "ffO4")
{
m_tensorell.set_chi(sigma);
}
}

template<class ContainerType0>
void set_chi( const SparseTensor<ContainerType0>& tau)
{
if (m_mode == "ff")
m_ell.set_chi(tau);
if (m_mode == "ffO4")
{
m_tensorell.set_chi(tau);
}
}

template<class ContainerType0>
void set_iota( const ContainerType0& sigma)
{
if (m_mode == "ffO4")
{
m_tensorell.set_iota(sigma);
}
}

void set_commute( bool commute) {m_commute = commute;}

bool get_commute() const {return m_commute;}

const Container& weights()const {
if (m_mode == "ffO4")
return  m_tensorell.weights();
else return  m_ell.weights();
}

const Container& precond()const {
if (m_mode == "ffO4")
return m_tensorell.precond();
else return m_ell.precond();
}

template<class ContainerType0, class ContainerType1>
void variation( const ContainerType0& phi, ContainerType1& varphi)
{
if (m_mode == "ff")
m_ell.variation(phi, varphi);
if (m_mode == "ffO4")
m_tensorell.variation(phi, varphi);
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

if (m_alpha == 0)
{
dg::blas1::scal( y, beta);
return;
}
if (m_mode == "df")
{
if (m_commute == false)
{
m_temp2_ex.extrapolate(m_temp2);
std::vector<unsigned> number = m_multi_g.solve(
m_multi_gamma, m_temp2, x, m_eps_gamma);
m_temp2_ex.update(m_temp2);

m_ell.symv(alpha, m_temp2, beta, y);
}
else
{
m_ell.symv(1.0, x, 0.0, m_temp);

m_temp2_ex.extrapolate(m_temp2);
std::vector<unsigned> number = m_multi_g.solve(
m_multi_gamma, m_temp2, m_temp, m_eps_gamma);
m_temp2_ex.update(m_temp2);

dg::blas1::axpby(alpha, m_temp2, beta, y);

}

}
if (m_mode == "ff" ) 
{
if (m_commute == false)
{
unsigned number = 0 ;
dg::apply( m_inv_sqrt, x, m_temp2);

m_ell.symv(1.0, m_temp2, 0.0, m_temp);

dg::apply( m_inv_sqrt, m_temp, m_temp2);
number++;

dg::blas1::axpby(alpha, m_temp2, beta, y);
}
else
{
}
}
if (m_mode == "ffO4")
{
if (m_commute == false)
{
m_temp2_ex.extrapolate(m_temp2);
std::vector<unsigned> number = m_multi_g.solve(
m_multi_gamma, m_temp2, x, m_eps_gamma);
m_temp2_ex.update(m_temp2);

m_tensorell.symv(1.0, m_temp2, 0.0, m_temp);

m_temp_ex.extrapolate(m_temp2);
number = m_multi_g.solve( m_multi_gamma, m_temp2,
m_temp, m_eps_gamma);
m_temp_ex.update(m_temp2);

dg::blas1::axpby(alpha, m_temp2, beta, y);
}
if (m_commute == true)
{
}
}
}

private:
dg::Elliptic<Geometry,  Matrix, Container> m_ell;
dg::mat::TensorElliptic<Geometry,  Matrix, Container> m_tensorell;

std::vector< dg::Helmholtz<Geometry,  Matrix, Container> > m_multi_gamma;
dg::MultigridCG2d<Geometry, Matrix, Container> m_multi_g;
dg::mat::MatrixSqrt<Container> m_inv_sqrt;
Container m_temp, m_temp2;
value_type  m_alpha;
std::vector<value_type> m_eps_gamma;
std::string m_mode;
dg::Extrapolation<Container> m_temp2_ex, m_temp_ex;
bool m_commute;


};

}  

template< class G, class M, class V>
struct TensorTraits< mat::PolCharge<G, M, V> >
{
using value_type      = get_value_type<V>;
using tensor_category = SelfMadeMatrixTag;
};
}  
