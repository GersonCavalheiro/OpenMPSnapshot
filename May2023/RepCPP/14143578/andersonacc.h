#pragma once

#include <functional>
#include "blas.h"
#include "topology/operator.h"

namespace dg{
namespace detail{

template<class ContainerType, class value_type>
void QRdelete1( std::vector<ContainerType>& Q, dg::Operator<value_type>& R, unsigned mMax)
{
for(unsigned i = 0; i<mMax-1;i++){
value_type temp = sqrt(R(i,i+1)*R(i,i+1)+R(i+1,i+1)*R(i+1,i+1));
value_type c = R(i,i+1)/temp;
value_type s = R(i+1,i+1)/temp;
R(i,i+1) = temp;
R(i+1,i+1) = 0;
if (i < mMax-2) {
for (unsigned j = i+2; j < mMax; j++){
temp = c * R(i,j) + s * R(i+1,j);
R(i+1,j) = - s * R(i,j) + c * R(i+1,j);
R(i,j) = temp;
}
}
dg::blas1::subroutine( [c,s]DG_DEVICE( double& qi, double& qip) {
double tmp = c*qi + s*qip;
qip = - s*qi + c*qip;
qi = tmp;
}, Q[i], Q[i+1]);
} 
for(unsigned i = 0; i<mMax-1;i++)
for(unsigned j = 0; j < mMax-1; j++)
R(i,j) = R(i,j+1);
return;
}

}




template<class ContainerType>
struct AndersonAcceleration
{
using value_type = get_value_type<ContainerType>;
using container_type = ContainerType; 
AndersonAcceleration() = default;

AndersonAcceleration(const ContainerType& copyable):
AndersonAcceleration( copyable, 0){}

AndersonAcceleration(const ContainerType& copyable, unsigned mMax ):
m_g_old( copyable), m_fval( copyable), m_f_old(copyable),
m_DG( mMax, copyable), m_Q( m_DG),
m_gamma( mMax, 0.),
m_R( mMax), m_mMax( mMax)
{
}


template<class ...Params>
void construct( Params&& ...ps)
{
*this = AndersonAcceleration( std::forward<Params>( ps)...);
}

const ContainerType& copyable() const{ return m_fval;}
void set_throw_on_fail( bool throw_on_fail){
m_throw_on_fail = throw_on_fail;
}


template<class MatrixType, class ContainerType0, class ContainerType1, class ContainerType2>
unsigned solve( MatrixType&& f, ContainerType0& x, const ContainerType1& b,
const ContainerType2& weights,
value_type rtol, value_type atol, unsigned max_iter,
value_type damping, unsigned restart, bool verbose);

private:
ContainerType m_g_old, m_fval, m_f_old;
std::vector<ContainerType> m_DG, m_Q;
std::vector<value_type> m_gamma;
dg::Operator<value_type> m_R;

unsigned m_mMax;
bool m_throw_on_fail = true;
};

template<class ContainerType>
template<class MatrixType, class ContainerType0, class ContainerType1, class ContainerType2>
unsigned AndersonAcceleration<ContainerType>::solve(
MatrixType&& func, ContainerType0& x, const ContainerType1& b, const ContainerType2& weights,
value_type rtol, value_type atol, unsigned max_iter,
value_type damping, unsigned restart,  bool verbose )
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
if (m_mMax == 0){
if(verbose)DG_RANK0 std::cout<< "No acceleration will occur" << std::endl;
}

unsigned mAA = 0;
value_type nrmb = sqrt( dg::blas2::dot( b, weights, b));
value_type tol = atol+rtol*nrmb;
if(verbose)DG_RANK0 std::cout << "Solve with mMax = "<<m_mMax<<" rtol = "
<<rtol<<" atol = "<<atol<< " tol = " << tol <<" max_iter =  "<<max_iter
<<" damping = "<<damping<<" restart = "<<restart<<std::endl;

ContainerType0& m_gval = x;
for(unsigned iter=0;iter < max_iter; iter++)
{
if ( restart != 0 && iter % (restart) == 0) {
mAA = 0;
if(verbose)DG_RANK0 std::cout << "Iter = " << iter << std::endl;
}

dg::apply( std::forward<MatrixType>(func), x, m_fval);
dg::blas1::axpby( -1., b, 1., m_fval); 
value_type res_norm = sqrt(dg::blas2::dot(m_fval,weights,m_fval));  

if(verbose)DG_RANK0 std::cout << "res_norm = " << res_norm << " Against tol = " << tol << std::endl;
if (res_norm <= tol){
if(verbose)DG_RANK0 std::cout << "Terminate with residual norm = " << res_norm << std::endl;
return iter+1;
}

dg::blas1::axpby(1.,x,-damping,m_fval,m_gval);                      
if( m_mMax == 0) continue;

if( iter == 0)
{
std::swap(m_fval,m_f_old);
dg::blas1::copy(m_gval,m_g_old);
continue;
}


if (mAA < m_mMax) {

dg::blas1::axpby(1.,m_fval,-1.,m_f_old, m_Q[mAA]);                 
dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[mAA]);        

} else {

std::rotate(m_DG.begin(), m_DG.begin() + 1, m_DG.end());  
dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[m_mMax-1]);     

detail::QRdelete1(m_Q,m_R,m_mMax);                      
dg::blas1::axpby(1.,m_fval,-1.,m_f_old, m_Q[m_mMax-1]);                 
mAA = m_mMax-1; 

}


for (unsigned j = 0; j < mAA; j++) {
m_R(j,mAA) = dg::blas2::dot(m_Q[j],weights,m_Q[mAA]);      

dg::blas1::axpby(-m_R(j,mAA),m_Q[j],1.,m_Q[mAA]);  
}
m_R(mAA,mAA) = sqrt(dg::blas2::dot(m_Q[mAA],weights,m_Q[mAA]));
dg::blas1::scal(m_Q[mAA], 1./m_R(mAA,mAA));


for(int i = (int)mAA; i>=0; i--){
m_gamma[i] = dg::blas2::dot(m_Q[i],weights,m_fval);
for(int j = i + 1; j < (int)mAA+1; j++){
m_gamma[i] = DG_FMA( -m_R(i,j), m_gamma[j], m_gamma[i]) ;
}
m_gamma[i] /= m_R(i,i);
}

std::swap(m_fval,m_f_old);
dg::blas1::copy(m_gval,m_g_old);

dg::blas2::gemv( -1., dg::asDenseMatrix( dg::asPointers(m_DG), mAA+1),
std::vector<value_type>{m_gamma.begin(), m_gamma.begin()+mAA+1},
1., x);

mAA++;
}
if( m_throw_on_fail)
{
throw dg::Fail( tol, Message(_ping_)
<<"After "<<max_iter<<" Anderson iterations with rtol "<<rtol<<" atol "<<atol<<" damping "<<damping<<" restart "<<restart);
}
return max_iter;

}

template<class ContainerType>
using FixedPointIteration = AndersonAcceleration<ContainerType>;

}
