#pragma once

#include "dg/algorithm.h"
#include "tridiaginv.h"


namespace dg {
namespace mat {


template<class value_type, class ExplicitRHS>
auto make_directODESolve( ExplicitRHS&& ode,
std::string tableau, value_type epsTimerel, value_type epsTimeabs,
unsigned& number, value_type t0 = 0., value_type t1 = 1.)
{
return [=, &num = number,
cap = std::tuple<ExplicitRHS>(std::forward<ExplicitRHS>(ode)),
rtol = epsTimerel, atol = epsTimeabs]
( const auto& x, auto& b) mutable
{
value_type reject_limit = 2;
dg::Adaptive<dg::ERKStep<std::decay_t<decltype(b)>>> adapt( tableau, x);
dg::AdaptiveTimeloop<std::decay_t<decltype(b)>> loop( adapt,
std::get<0>(cap), dg::pid_control, dg::l2norm, rtol, atol,
reject_limit);
loop.integrate( t0, x, t1, b);
num = adapt.nsteps();
};
}


template< class Container>
struct InvSqrtODE
{
public:
using container_type = Container;
using value_type = dg::get_value_type<Container>;
InvSqrtODE() {};


template<class MatrixType>
InvSqrtODE( MatrixType& A, const Container& copyable)
{
m_helper = copyable;
m_A = [&]( const Container& x, Container& y){
return dg::apply( A, x, y);
};
m_yp_ex.set_max(3, copyable);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = SqrtODE( std::forward<Params>( ps)...);
}

const value_type& time() const{ return m_time;}

auto make_operator() const{
return [&t = m_time, &A = m_A](  const Container& x, Container& y)
{
dg::blas2::symv(A, x, y);
dg::blas1::axpby( t, x, (1.-t), y);
};
}
template<class MatrixType>
void set_inverse_operator( const MatrixType& OpInv ) {
m_Ainv = OpInv;
}

void operator()(value_type t, const Container& y, Container& yp)
{
m_time = t;
dg::blas2::symv(m_A, y, m_helper);
dg::blas1::axpby(0.5, y, -0.5, m_helper);

m_yp_ex.extrapolate(t, yp);
dg::blas2::symv( m_Ainv, m_helper, yp);
m_yp_ex.update(t, yp);
}
private:
Container m_helper;
std::function<void(const Container&, Container&)> m_A, m_Ainv;
value_type m_time;
dg::Extrapolation<Container> m_yp_ex;
};



template< class Matrix, class Preconditioner, class Container>
InvSqrtODE<Container> make_inv_sqrtodeCG( Matrix& A, const Preconditioner& P,
const Container& weights, dg::get_value_type<Container> epsCG)
{
InvSqrtODE<Container> sqrtode( A, weights);
dg::PCG<Container> pcg( weights, 10000);
auto op = sqrtode.make_operator();
sqrtode.set_inverse_operator( [ = ]( const auto& x, auto& y) mutable
{
pcg.solve( op, y, x, P, weights, epsCG);
});
return sqrtode;
}

template< class Matrix, class Container>
InvSqrtODE<Container> make_inv_sqrtodeTri( const Matrix& TH, const Container&
copyable)
{
InvSqrtODE<Container> sqrtode( TH, copyable);
sqrtode.set_inverse_operator( [ &TH = TH, &t = sqrtode.time() ]
( const auto& x, auto& y) mutable
{
dg::mat::compute_Tinv_y( TH, y, x, (1.-t), t);
});
return sqrtode;
}


template<class MatrixType>
auto make_expode( MatrixType& A)
{
return [&]( auto t, const auto& y, auto& yp) mutable
{
dg::blas2::symv( A, y, yp);
};
}


template<class MatrixType>
auto make_besselI0ode( MatrixType& A)
{
return [&m_A = A]( auto t, const auto& z, auto& zp) mutable
{
dg::blas2::symv(m_A, z[0], zp[0]);
dg::blas2::symv(m_A, zp[0], zp[1]);
dg::blas1::axpby(-1./t, z[1], 1.0, zp[1]);
dg::blas1::copy(z[0],zp[0]);

};
}


} 
} 
