#pragma once

#include "dg/algorithm.h"
#include "lanczos.h"

namespace dg{
namespace mat{



template<class ContainerType>
struct MatrixSqrt
{
using container_type = ContainerType;
using value_type = dg::get_value_type<ContainerType>;

MatrixSqrt() = default;


template<class MatrixType>
MatrixSqrt(  MatrixType& A, int exp,
const ContainerType& weights, value_type eps_rel,
value_type nrmb_correction  = 1.,
unsigned max_iter = 500, unsigned cauchy_steps = 40
) : m_weights(weights),
m_exp(exp), m_cauchy( cauchy_steps), m_eps(eps_rel),
m_abs(nrmb_correction)
{
m_A = [&]( const ContainerType& x, ContainerType& y){
return dg::apply( A, x, y);
};
m_lanczos.construct( weights, max_iter);
dg::mat::UniversalLanczos<ContainerType> eigen( weights, 20);
auto T = eigen.tridiag( A, weights, weights);
m_EVs = dg::mat::compute_extreme_EV( T);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = MatrixSqrt( std::forward<Params>( ps)...);
}
unsigned get_iter() const{return m_number;}


void set_benchmark( bool benchmark, std::string message = "SQRT"){
m_benchmark = benchmark;
m_message = message;
}


template<class ContainerType0, class ContainerType1>
void operator()( const ContainerType0 b, ContainerType1& x)
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
dg::Timer t;
t.tic();
auto func = make_SqrtCauchyEigen_Te1( m_exp, m_EVs, m_cauchy);
m_number = m_lanczos.solve( x, func, m_A, b, m_weights, m_eps, m_abs,
"universal", 1., 2);
t.toc();
if( m_benchmark)
DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<m_number<<","<<m_cauchy<<"} iterations took "<<t.diff()<<"s\n";
}
private:
UniversalLanczos<ContainerType> m_lanczos;
ContainerType m_weights;
std::function< void( const ContainerType&, ContainerType&)> m_A;
std::array<value_type, 2> m_EVs;
int m_exp;
unsigned m_number, m_cauchy;
value_type m_eps, m_abs;
bool m_benchmark = true;
std::string m_message = "SQRT";

};


template<class ContainerType>
struct MatrixFunction
{
using container_type = ContainerType;
using value_type = dg::get_value_type<ContainerType>;

MatrixFunction() = default;


template<class MatrixType>
MatrixFunction(  MatrixType& A,
const ContainerType& weights, value_type eps_rel,
value_type nrmb_correction  = 1.,
unsigned max_iter = 500,
std::function<value_type(value_type)> f_inner = [](value_type x){return x;}
) : m_weights(weights),
m_f_inner(f_inner), m_eps(eps_rel),
m_abs(nrmb_correction)
{
m_A = [&]( const ContainerType& x, ContainerType& y){
return dg::apply( A, x, y);
};
m_lanczos.construct( weights, max_iter);
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = MatrixFunction( std::forward<Params>( ps)...);
}
unsigned get_iter() const{return m_number;}


void set_benchmark( bool benchmark, std::string message = "Function"){
m_benchmark = benchmark;
m_message = message;
}


template<class UnaryOp, class ContainerType0, class ContainerType1>
void operator()( UnaryOp f_outer, const ContainerType0 b, ContainerType1& x)
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
dg::Timer t;
t.tic();
auto func = make_FuncEigen_Te1( [&](value_type x) {return f_outer(m_f_inner(x));});
m_number = m_lanczos.solve( x, func, m_A, b, m_weights, m_eps, m_abs,
"universal", 1., 2);
t.toc();
if( m_benchmark)
DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<m_number<<"} iterations took "<<t.diff()<<"s\n";
}
private:
UniversalLanczos<ContainerType> m_lanczos;
ContainerType m_weights;
std::function< void( const ContainerType&, ContainerType&)> m_A;
std::array<value_type, 2> m_EVs;
std::function<value_type(value_type)> m_f_inner;
unsigned m_number;
value_type m_eps, m_abs;
bool m_benchmark = true;
std::string m_message = "Function";

};
}
}
