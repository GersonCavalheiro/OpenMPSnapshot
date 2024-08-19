#pragma once
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

#include "dg/algorithm.h"
#include "tridiaginv.h"
#include "matrixfunction.h"



namespace dg{
namespace mat{


template< class ContainerType>
class MCG
{
public:
using value_type = dg::get_value_type<ContainerType>; 
using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
using HVec = dg::HVec;
MCG(){}

MCG( const ContainerType& copyable, unsigned max_iterations)
{
m_ap = m_p = m_r = copyable;
m_max_iter = max_iterations;
set_iter( max_iterations);
}

void set_max( unsigned new_max) {
m_max_iter = new_max;
set_iter( new_max);
}

unsigned get_max() const {return m_max_iter;}

void set_verbose( bool verbose){ m_verbose = verbose;}

value_type get_bnorm() const{return m_bnorm;}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = MCG( std::forward<Params>( ps)...);
}
unsigned get_iter() const {return m_iter;}

template< class MatrixType, class DiaMatrixType, class ContainerType0,
class ContainerType1, class ContainerType2>
void Ry( MatrixType&& A, const DiaMatrixType& T,
const ContainerType0& y, ContainerType1& x,
const ContainerType2& b)
{
dg::blas1::copy(0., x);

dg::blas1::copy( b, m_r);

dg::blas1::copy( m_r, m_p );

dg::blas1::axpby( y[0], m_r, 1., x); 
for ( unsigned i=0; i<y.size()-1; i++)
{
dg::blas2::symv( std::forward<MatrixType>(A), m_p, m_ap);
value_type alphainv = i==0 ? T.values( i,1) :
T.values(i,1) + T.values( i-1,2);
value_type beta = -T.values( i,2)/alphainv;
dg::blas1::axpby( -1./alphainv, m_ap, 1., m_r);
dg::blas1::axpby(1., m_r, beta, m_p );
dg::blas1::axpby( y[i+1], m_r, 1., x); 
}
}

template< class MatrixType, class ContainerType0, class ContainerType1>
const HDiaMatrix& operator()( MatrixType&& A, const ContainerType0& b,
const ContainerType1& weights, value_type eps = 1e-12,
value_type nrmb_correction = 1., value_type res_fac = 1.)
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
value_type nrmzr_old = dg::blas2::dot( b, weights, b);
value_type nrmb = sqrt(nrmzr_old);
m_bnorm = nrmb;
if( m_verbose)
{
DG_RANK0 std::cout << "# Norm of b  "<<nrmb <<"\n";
DG_RANK0 std::cout << "# Res factor "<<res_fac <<"\n";
DG_RANK0 std::cout << "# Residual errors: \n";
}
if( nrmb == 0)
{
set_iter(1);
return m_TH;
}
dg::blas1::copy( b, m_r);
dg::blas1::copy( m_r, m_p );

value_type alpha = 0, beta = 0, nrmzr_new = 0, alpha_old = 0., beta_old = 0.;
for( unsigned i=0; i<m_max_iter; i++)
{
alpha_old = alpha, beta_old = beta;
dg::blas2::symv( std::forward<MatrixType>(A), m_p, m_ap);
alpha = nrmzr_old /dg::blas2::dot( m_p, weights, m_ap);
dg::blas1::axpby( -alpha, m_ap, 1., m_r);
nrmzr_new = dg::blas2::dot( m_r, weights, m_r);
beta = nrmzr_new/nrmzr_old;
if(m_verbose)
{
DG_RANK0 std::cout << "# ||r||_W = " << sqrt(nrmzr_new) << "\tat i = " << i << "\n";
}
if( i == 0)
{
m_TH.values(i,0) = 0.;
m_TH.values(i,1) = 1./alpha;
m_TH.values(i,2) = -beta/alpha;
}
else
{
m_TH.values(i,0) = -1./alpha_old;
m_TH.values(i,1) =  1./alpha + beta_old/alpha_old;
m_TH.values(i,2) = -beta/alpha;
}
if( res_fac*sqrt( nrmzr_new)
< eps*(nrmb + nrmb_correction))
{
set_iter(i+1);
break;
}
dg::blas1::axpby(1., m_r, beta, m_p );
nrmzr_old=nrmzr_new;
}

return m_TH;
}

HVec make_e1( ) {
HVec e1H(m_iter, 0.);
e1H[0] = 1.;
return e1H;
}
private:
void set_iter( unsigned new_iter) {
m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
m_TH.diagonal_offsets[0] = -1;
m_TH.diagonal_offsets[1] =  0;
m_TH.diagonal_offsets[2] =  1;
m_iter = new_iter;
}
ContainerType m_r, m_ap, m_p;
unsigned m_max_iter, m_iter;
HDiaMatrix m_TH;
bool m_verbose = false;
value_type m_bnorm = 0.;
};


template<class Container>
class MCGFuncEigen
{
public:
using value_type = dg::get_value_type<Container>;
using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
using HArray2d = cusp::array2d< value_type, cusp::host_memory>;
using HArray1d = cusp::array1d< value_type, cusp::host_memory>;
using HVec = dg::HVec;
MCGFuncEigen(){}

MCGFuncEigen( const Container& copyable, unsigned max_iterations)
{
m_e1H.assign(max_iterations, 0.);
m_e1H[0] = 1.;
m_yH.assign(max_iterations, 1.);
m_mcg.construct(copyable, max_iterations);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = MCGFuncEigen( std::forward<Params>( ps)...);
}

template < class MatrixType, class ContainerType0, class ContainerType1 ,
class ContainerType2, class UnaryOp>
unsigned operator()(ContainerType0& x, UnaryOp f,
MatrixType&& A, const ContainerType1& b,
const ContainerType2& weights, value_type eps,
value_type nrmb_correction, value_type res_fac )
{
if( sqrt(dg::blas2::dot(b, weights, b)) == 0)
{
dg::blas1::copy( b, x);
return 0;
}
m_TH = m_mcg(std::forward<MatrixType>(A), b, weights, eps,
nrmb_correction, res_fac);
unsigned iter = m_mcg.get_iter();
m_alpha.resize(iter);
m_delta.resize(iter,1.);
m_beta.resize(iter-1);
m_evals.resize(iter);
m_evecs.resize(iter,iter);
m_e1H.resize(iter, 0.);
m_e1H[0] = 1.;
m_yH.resize(iter);
for(unsigned i = 0; i<iter; i++)
{
m_alpha[i] = m_TH.values(i,1);
if (i<iter-1) {
if      (m_TH.values(i,2) > 0.) m_beta[i] =  sqrt(m_TH.values(i,2)*m_TH.values(i+1,0)); 
else if (m_TH.values(i,2) < 0.) m_beta[i] = -sqrt(m_TH.values(i,2)*m_TH.values(i+1,0)); 
else m_beta[i] = 0.;
}
if (i>0) m_delta[i] = m_delta[i-1]*sqrt(m_TH.values(i,0)/m_TH.values(i-1,2));
}
cusp::lapack::stev(m_alpha, m_beta, m_evals, m_evecs);
cusp::convert(m_evecs, m_EH);
cusp::transpose(m_EH, m_EHt);
dg::blas1::pointwiseDivide(m_e1H, m_delta, m_e1H);
dg::blas2::symv(m_EHt, m_e1H, m_yH);
dg::blas1::transform(m_evals, m_e1H, [f] (double x){
try{
return f(x);
}
catch(boost::exception& e) 
{
return 0.;
}
});
dg::blas1::pointwiseDot(m_e1H, m_yH, m_e1H);
dg::blas2::symv(m_EH, m_e1H, m_yH);
dg::blas1::pointwiseDot(m_yH, m_delta, m_yH);
m_mcg.Ry(std::forward<MatrixType>(A), m_TH, m_yH, x, b);

return iter;
}
private:
HVec m_e1H, m_yH;
HDiaMatrix m_TH;
HCooMatrix m_EH, m_EHt;
HArray2d m_evecs;
HArray1d m_alpha, m_beta, m_delta, m_evals;
dg::mat::MCG< Container > m_mcg;

};
} 
} 

