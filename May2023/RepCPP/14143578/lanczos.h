#pragma once
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

#include "dg/algorithm.h"
#include "tridiaginv.h"
#include "matrixfunction.h"




namespace dg{
namespace mat{


template< class ContainerType >
class UniversalLanczos
{
public:
using value_type = get_value_type<ContainerType>; 
using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
using HVec = dg::HVec;
UniversalLanczos(){}

UniversalLanczos( const ContainerType& copyable, unsigned max_iterations)
{
m_v = m_vp = m_vm = copyable;
m_max_iter = max_iterations;
m_iter = max_iterations;
set_iter( max_iterations);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = UniversalLanczos( std::forward<Params>( ps)...);
}

void set_max( unsigned new_max) {
m_max_iter = new_max;
set_iter( new_max);
}

unsigned get_max() const {return m_max_iter;}

void set_verbose( bool verbose){ m_verbose = verbose;}

value_type get_bnorm() const{return m_bnorm;}


template < class MatrixType, class ContainerType0, class ContainerType1,
class ContainerType2, class FuncTe1>
unsigned solve(ContainerType0& x, FuncTe1 f,
MatrixType&& A, const ContainerType1& b,
const ContainerType2& weights, value_type eps,
value_type nrmb_correction = 1.,
std::string error_norm = "universal",
value_type res_fac = 1.,
unsigned q = 1 )
{
tridiag( f, std::forward<MatrixType>(A), b, weights, eps,
nrmb_correction, error_norm, res_fac, q);
if( "residual" == error_norm)
m_yH = f( m_TH);
normMbVy(std::forward<MatrixType>(A), m_TH, m_yH, x, b,
m_bnorm);
return m_iter;
}

template< class MatrixType, class ContainerType0, class ContainerType1>
const HDiaMatrix& tridiag( MatrixType&& A, const ContainerType0& b,
const ContainerType1& weights, value_type eps = 1e-12,
value_type nrmb_correction = 1.,
std::string error_norm = "universal", value_type res_fac = 1.,
unsigned q = 1
)
{
auto op = make_Linear_Te1( -1);
tridiag( op, std::forward<MatrixType>(A), b, weights, eps,
nrmb_correction, error_norm, res_fac, q);
return m_TH;
}

unsigned get_iter() const {return m_iter;}
private:


template< class MatrixType, class DiaMatrixType, class ContainerType0,
class ContainerType1,class ContainerType2>
void normMbVy( MatrixType&& A,
const DiaMatrixType& T,
const ContainerType0& y,
ContainerType1& x,
const ContainerType2& b, value_type bnorm)
{
dg::blas1::copy(0., x);
if( 0 == bnorm )
{
return;
}
dg::blas1::axpby(1./bnorm, b, 0.0, m_v); 
dg::blas1::copy(0., m_vm);
unsigned less_iter = 0;
for( unsigned i=0; i<y.size(); i++)
if( y[i] != 0)
less_iter = i+1;
dg::blas1::axpby( y[0]*bnorm, m_v, 1., x); 
for ( unsigned i=0; i<less_iter-1; i++)
{
dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
dg::blas1::axpbypgz(
-T.values(i,0)/T.values(i,2), m_vm,
-T.values(i,1)/T.values(i,2), m_v,
1.0/T.values(i,2), m_vp);
dg::blas1::axpby( y[i+1]*bnorm, m_vp, 1., x); 
m_vm.swap( m_v);
m_v.swap( m_vp);

}
}
template < class MatrixType, class ContainerType1,
class ContainerType2, class UnaryOp>
void tridiag(UnaryOp f,
MatrixType&& A, const ContainerType1& b,
const ContainerType2& weights, value_type eps,
value_type nrmb_correction,
std::string error_norm = "residual",
value_type res_fac = 1.,
unsigned q = 1 )
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
m_bnorm = sqrt(dg::blas2::dot(b, weights, b));
if( m_verbose)
{
DG_RANK0 std::cout << "# Norm of b  "<<m_bnorm <<"\n";
DG_RANK0 std::cout << "# Res factor "<<res_fac <<"\n";
DG_RANK0 std::cout << "# Residual errors: \n";
}
if( m_bnorm == 0)
{
set_iter(1);
return;
}
value_type residual;
dg::blas1::axpby(1./m_bnorm, b, 0.0, m_v); 
value_type betaip = 0.;
value_type alphai = 0.;
for( unsigned i=0; i<m_max_iter; i++)
{
m_TH.values(i,0) =  betaip; 
dg::blas2::symv(std::forward<MatrixType>(A), m_v, m_vp);
dg::blas1::axpby(-betaip, m_vm, 1.0, m_vp);  
alphai  = dg::blas2::dot(m_vp, weights, m_v);
m_TH.values(i,1) = alphai;
dg::blas1::axpby(-alphai, m_v, 1.0, m_vp);
betaip = sqrt(dg::blas2::dot(m_vp, weights, m_vp));
if (betaip == 0)
{
if( m_verbose)
DG_RANK0 std::cout << "beta["<<i+1 <<"]=0 encountered\n";
set_iter(i+1);
break;
}
m_TH.values(i,2) = betaip;  

value_type xnorm = 0.;
if( "residual" == error_norm)
{
residual = compute_residual_error( m_TH, i)*m_bnorm;
xnorm = m_bnorm;
}
else
{
if( i>=q &&(  (i<=10) || (i>10 && i%10 == 0) ))
{
residual = compute_universal_error( m_TH, i, q, f,
m_yH)*m_bnorm;
xnorm = dg::fast_l2norm( m_yH)*m_bnorm;
}
else
{
residual = 1e10;
xnorm = m_bnorm;
}
}
if( m_verbose)
DG_RANK0 std::cout << "# ||r||_W = " << residual << "\tat i = " << i << "\n";
if (res_fac*residual< eps*(xnorm + nrmb_correction) )
{
set_iter(i+1);
break;
}
dg::blas1::scal(m_vp, 1./betaip);
m_vm.swap(m_v);
m_v.swap( m_vp);
set_iter( m_max_iter);
}
}
value_type compute_residual_error( const HDiaMatrix& TH, unsigned iter)
{
value_type T1 = compute_Tinv_m1( TH, iter+1);
return TH.values(iter,2)*fabs(T1); 
}
template<class UnaryOp>
value_type compute_universal_error( const HDiaMatrix& TH, unsigned iter,
unsigned q, UnaryOp f, HVec& yH)
{
unsigned new_iter = iter + 1 + q;
set_iter( iter+1);
HDiaMatrix THtilde( new_iter, new_iter, 3*new_iter-2, 3);
THtilde.diagonal_offsets[0] = -1;
THtilde.diagonal_offsets[1] =  0;
THtilde.diagonal_offsets[2] =  1;
for( unsigned u=0; u<iter+1; u++)
{
THtilde.values(u,0) = TH.values(u,0);
THtilde.values(u,1) = TH.values(u,1);
THtilde.values(u,2) = TH.values(u,2);
}
for( unsigned u=1; u<=q; u++)
{
THtilde.values( iter+u, 0) = u==1 ? TH.values(iter,2) :
TH.values( iter+1-u, 1);
THtilde.values( iter+u, 1) = TH.values( iter-u, 1);
THtilde.values( iter+u, 2) = TH.values( iter-u, 0);
}
yH = f( TH);
HVec yHtilde = f( THtilde);
for( unsigned u=0; u<yH.size(); u++)
yHtilde[u] -= yH[u];
value_type norm = dg::fast_l2norm( yHtilde);
return norm;
}

void set_iter( unsigned new_iter) {
m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
m_TH.diagonal_offsets[0] = -1;
m_TH.diagonal_offsets[1] =  0;
m_TH.diagonal_offsets[2] =  1;
m_iter = new_iter;
}
ContainerType  m_v, m_vp, m_vm;
HDiaMatrix m_TH;
HVec m_yH;
unsigned m_iter, m_max_iter;
bool m_verbose = false;
value_type m_bnorm = 0.;
};


} 
} 

