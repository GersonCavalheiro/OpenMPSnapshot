#pragma once
#include "pcg.h"

namespace dg{
namespace detail{
template<class Implicit, class Solver>
struct Adaptor
{
Adaptor( Implicit& im, Solver& solver) : m_im(im), m_solver(solver){}
template<class ContainerType, class value_type>
void operator()( value_type t, const ContainerType& x, ContainerType& y)
{
m_im( t,x,y);
}
template<class ContainerType, class value_type>
void operator()( value_type alpha, value_type t, ContainerType& y, const ContainerType& yp)
{
m_solver.solve( alpha, m_im, t, y, yp);
}
private:
Implicit& m_im;
Solver& m_solver;
};
}


template<class ContainerType>
struct DefaultSolver
{
using container_type = ContainerType;
using value_type = get_value_type<ContainerType>;
DefaultSolver(){}

template<class Implicit>
DefaultSolver( Implicit& im, const ContainerType& copyable,
unsigned max_iter, value_type eps): m_max_iter(max_iter)
{
m_im = [&im = im]( value_type t, const ContainerType& y, ContainerType&
yp) mutable
{
im( t, y, yp);
};
m_solve = [ &weights = im.weights(), &precond = im.precond(), pcg =
dg::PCG<ContainerType>( copyable, max_iter), eps = eps ]
( const std::function<void( const ContainerType&,ContainerType&)>&
wrapper, ContainerType& y, const ContainerType& ys) mutable
{
return pcg.solve( wrapper, y, ys, precond, weights, eps);
};
}
template<class ...Params>
void construct( Params&& ...ps)
{
*this = DefaultSolver( std::forward<Params>( ps)...);
}
void set_benchmark( bool benchmark){ m_benchmark = benchmark;}

void operator()( value_type alpha, value_type time, ContainerType& y, const
ContainerType& ys)
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
auto wrapper = [a = alpha, t = time, &i = m_im]( const auto& x, auto& y){
i( t, x, y);
dg::blas1::axpby( 1., x, -a, y);
};
Timer ti;
if(m_benchmark) ti.tic();
dg::blas1::copy( ys, y); 
unsigned number = m_solve( wrapper, y, ys);
if( m_benchmark)
{
ti.toc();
DG_RANK0 std::cout << "# of pcg iterations time solver: "
<<number<<"/"<<m_max_iter<<" took "<<ti.diff()<<"s\n";
}
}
private:
std::function<void( value_type, const ContainerType&, ContainerType&)>
m_im;
std::function< unsigned ( const std::function<void( const
ContainerType&,ContainerType&)>&, ContainerType&,
const ContainerType&)> m_solve;
unsigned m_max_iter;
bool m_benchmark = true;
};

}
