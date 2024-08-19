#pragma once
#include <cmath>

#include <boost/math/special_functions.hpp> 
#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/lapack/lapack.h>
#include "dg/algorithm.h"

#include "functors.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"


namespace dg {
namespace mat {


template<class UnaryOp>
auto make_FuncEigen_Te1( UnaryOp f)
{
return [f]( const auto& T)
{
using value_type = typename std::decay_t<decltype(T)>::value_type;
unsigned iter = T.num_rows;
cusp::array2d< value_type, cusp::host_memory> evecs(iter,iter);
cusp::array1d< value_type, cusp::host_memory> evals(iter);
dg::HVec e1H(iter,0.), yH(e1H);
e1H[0] = 1.;
yH.resize( iter);
cusp::lapack::stev(T.values.column(1), T.values.column(2),
evals, evecs, 'V');
cusp::coo_matrix<int, value_type, cusp::host_memory> EH, EHt;
cusp::convert(evecs, EH);
cusp::transpose(EH, EHt);
dg::blas2::symv(EHt, e1H, yH);
dg::blas1::transform(evals, e1H, [f] (double x){
try{
return f(x);
}
catch(boost::exception& e) 
{
return 0.;
}
});
dg::blas1::pointwiseDot(e1H, yH, e1H);
dg::blas2::symv(EH, e1H, yH);
return yH;
};
}


template< class value_type>
auto make_SqrtCauchy_Te1( int exp, std::array<value_type,2> EVs, unsigned stepsCauchy)
{
return [=]( const auto& T)
{
unsigned size = T.num_rows;
thrust::host_vector<value_type> e1H(size, 0.), yH(e1H);
e1H[0] = 1.;

dg::mat::SqrtCauchyInt<HVec> cauchysqrtH( e1H);
auto wTinv = [&w = cauchysqrtH.w(), &T = T]( const auto& y, auto& x)
{
dg::mat::compute_Tinv_y( T, x, y, 1., w*w);
};
cauchysqrtH(T, wTinv, e1H, yH, EVs, stepsCauchy, exp);
return yH;
};
}


template< class value_type>
auto make_SqrtCauchyEigen_Te1( int exp, std::array<value_type,2> EVs, unsigned stepsCauchy)
{
std::function< value_type(value_type)> func = dg::SQRT<value_type>();
if( exp < 0)
func = [](value_type x){return 1./sqrt(x);};

auto eigen = make_FuncEigen_Te1( func);
auto cauchy = make_SqrtCauchy_Te1( exp, EVs, stepsCauchy);
return [=]( const auto& T)
{
unsigned size = T.num_rows;
dg::HVec yH;
if ( size < 40)
yH = eigen( T);
else
yH = cauchy(T);
return yH;
};
}



template< class value_type>
auto make_SqrtODE_Te1( int exp, std::string tableau, value_type rtol,
value_type atol, unsigned& number)
{
return [=, &num = number](const auto& T)
{
unsigned size = T.num_rows;
HVec e1H(size, 0), yH(e1H);
e1H[0] = 1.;
auto inv_sqrt = make_inv_sqrtodeTri( T, e1H);
auto sqrtHSolve =  make_directODESolve( inv_sqrt,
tableau, rtol, atol, num);
dg::apply( sqrtHSolve, e1H, yH);
if( exp >= 0 )
{
dg::apply( T, yH, e1H);
return e1H;
}
return yH;
};
}


inline static auto make_Linear_Te1( int exp)
{
return [= ](const auto& T)
{
unsigned size = T.num_rows;
HVec e1H(size, 0), yH(e1H);
e1H[0] = 1.;
if( exp < 0)
compute_Tinv_y( T, yH, e1H);
else
dg::blas2::symv( T, e1H, yH);
return yH;
};
}

} 
} 
