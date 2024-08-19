#pragma once

#include "backend/exceptions.h"
#include "backend/memory.h"
#include "topology/fast_interpolation.h"
#include "topology/interpolation.h"
#include "blas.h"
#include "pcg.h"
#include "chebyshev.h"
#include "eve.h"
#include "backend/timer.h"
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif

namespace dg
{



template<class Geometry, class Matrix, class Container>
struct NestedGrids
{
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
NestedGrids(): m_stages(0), m_grids(0), m_inter(0), m_project(0){}

template<class ...ContainerParams>
NestedGrids( const Geometry& grid, const unsigned stages, ContainerParams&& ...ps):
m_stages(stages),
m_grids( stages),
m_x( stages)
{
if(stages < 1 )
throw Error( Message(_ping_)<<" There must be minimum 1 stage in nested Grids construction! You gave " << stages);
m_grids[0].reset( grid);

for(unsigned u=1; u<stages; u++)
{
m_grids[u] = m_grids[u-1]; 
m_grids[u]->multiplyCellNumbers(0.5, 0.5);
}

m_inter.resize(    stages-1);
m_project.resize(  stages-1);
for(unsigned u=0; u<stages-1; u++)
{
m_project[u].construct( dg::create::fast_projection(*m_grids[u], 1,
2, 2), std::forward<ContainerParams>(ps)...);
m_inter[u].construct( dg::create::fast_interpolation(*m_grids[u+1],
1, 2, 2), std::forward<ContainerParams>(ps)...);
}
for( unsigned u=0; u<m_stages; u++)
m_x[u] = dg::construct<Container>( dg::evaluate( dg::zero,
*m_grids[u]), std::forward<ContainerParams>(ps)...);
m_w = m_r = m_b = m_x;

}

template<class ...Params>
void construct( Params&& ...ps)
{
*this = NestedGrids( std::forward<Params>( ps)...);
}

template<class ContainerType0>
void project( const ContainerType0& src, std::vector<ContainerType0>& out) const
{
dg::blas1::copy( src, out[0]);
for( unsigned u=0; u<m_stages-1; u++)
dg::blas2::gemv( m_project[u], out[u], out[u+1]);
}


template<class ContainerType0>
std::vector<ContainerType0> project( const ContainerType0& src) const
{
std::vector<Container> out( m_x);
project( src, out);
return out;

}
const Container& copyable() const {return m_x[0];}

unsigned stages()const{return m_stages;}
unsigned num_stages()const{return m_stages;}

const Geometry& grid( unsigned stage) const {
return *(m_grids[stage]);
}

const MultiMatrix<Matrix, Container>& interpolation( unsigned stage) const
{
return m_inter[stage];
}
const MultiMatrix<Matrix, Container>& projection( unsigned stage) const
{
return m_project[stage];
}
Container& x(unsigned stage){ return m_x[stage];}
const Container& x(unsigned stage) const{ return m_x[stage];}
Container& r(unsigned stage){ return m_r[stage];}
const Container& r(unsigned stage) const{ return m_r[stage];}
Container& b(unsigned stage){ return m_b[stage];}
const Container& b(unsigned stage) const{ return m_b[stage];}
Container& w(unsigned stage){ return m_w[stage];}
const Container& w(unsigned stage) const{ return m_w[stage];}

private:
unsigned m_stages;
std::vector< dg::ClonePtr< Geometry> > m_grids;
std::vector< MultiMatrix<Matrix, Container> >  m_inter;
std::vector< MultiMatrix<Matrix, Container> >  m_project;
std::vector< Container> m_x, m_r, m_b, m_w;
};


template<class MatrixType0, class ContainerType0, class ContainerType1, class MatrixType1, class NestedGrids>
void nested_iterations(
std::vector<MatrixType0>& ops, ContainerType0& x, const ContainerType1& b,
std::vector<MatrixType1>& inverse_ops, NestedGrids& nested_grids)
{
NestedGrids& nested = nested_grids;
dg::apply(ops[0], x, nested.r(0));
dg::blas1::axpby(1., b, -1., nested.r(0));
dg::blas1::copy( x, nested.x(0));
for( unsigned u=0; u<nested.stages()-1; u++)
{
dg::blas2::gemv( nested.projection(u), nested.r(u), nested.r(u+1));
dg::blas2::gemv( nested.projection(u), nested.x(u), nested.x(u+1));
dg::blas2::symv( ops[u+1], nested.x(u+1), nested.b(u+1));
dg::blas1::axpby( 1., nested.b(u+1), 1., nested.r(u+1), nested.b(u+1));
dg::blas1::copy( nested.x(u+1), nested.w(u+1)); 
}

for( unsigned u=nested.stages()-1; u>0; u--)
{
try{
dg::apply( inverse_ops[u],  nested.b(u), nested.x(u));
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<u<<" of nested iterations");
throw;
}
dg::blas1::axpby( 1., nested.x(u), -1., nested.w(u), nested.x(u) );
dg::blas2::symv( 1., nested.interpolation(u-1), nested.x(u), 1.,
nested.x(u-1));
}
dg::blas1::copy( nested.x(0), x);
try{
dg::apply(inverse_ops[0], b, x);
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on stage 0 of nested iterations");
throw;
}
}


template<class NestedGrids, class MatrixType0, class MatrixType1, class MatrixType2>
void multigrid_cycle(
std::vector<MatrixType0>& ops,
std::vector<MatrixType1>& inverse_ops_down, 
std::vector<MatrixType2>& inverse_ops_up, 
NestedGrids& nested_grids, unsigned gamma, unsigned p)
{
NestedGrids& nested = nested_grids;

try{
dg::apply( inverse_ops_down[p], nested.b(p), nested.x(p));
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on pre-smoothing stage "<<p<<" of multigrid cycle");
throw;
}
dg::apply( ops[p], nested.x(p), nested.r(p));
dg::blas1::axpby( 1., nested.b(p), -1., nested.r(p));
dg::blas2::symv( nested.projection(p), nested.r(p), nested.r(p+1));
dg::blas2::symv( nested.projection(p), nested.x(p), nested.x(p+1));
dg::blas2::symv( ops[p+1], nested.x(p+1), nested.b(p+1));
dg::blas1::axpby( 1., nested.r(p+1), 1., nested.b(p+1));
dg::blas1::copy( nested.x(p+1), nested.w(p+1));
if( p+1 == nested.stages()-1)
{
try{
dg::apply( inverse_ops_up[p+1], nested.b(p+1), nested.x(p+1));
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<p+1<<" of multigrid cycle");
throw;
}
}
else
{
for( unsigned u=0; u<gamma; u++)
{
multigrid_cycle( ops, inverse_ops_down, inverse_ops_up,
nested, gamma, p+1);
}
}

dg::blas1::axpby( 1., nested.x(p+1), -1., nested.w(p+1));
dg::blas2::symv( 1., nested.interpolation(p), nested.w(p+1), 1., nested.x(p));
try{
dg::apply(inverse_ops_up[p], nested.b(p), nested.x(p));
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on post-smoothing stage "<<p<<" of multigrid cycle");
throw;
}
}


template<class MatrixType0, class MatrixType1, class MatrixType2, class NestedGrids, class ContainerType0, class ContainerType1>
void full_multigrid(
std::vector<MatrixType0>& ops, ContainerType0& x, const ContainerType1& b,
std::vector<MatrixType1>& inverse_ops_down, 
std::vector<MatrixType2>& inverse_ops_up, 
NestedGrids& nested_grids, unsigned gamma, unsigned mu)
{
NestedGrids& nested = nested_grids;
dg::apply(ops[0], x, nested.r(0));
dg::blas1::axpby(1., b, -1., nested.r(0));
dg::blas1::copy( x, nested.x(0));
for( unsigned u=0; u<nested.stages()-1; u++)
{
dg::blas2::gemv( nested.projection(u), nested.r(u), nested.r(u+1));
dg::blas2::gemv( nested.projection(u), nested.x(u), nested.x(u+1));
dg::blas2::symv( ops[u+1], nested.x(u+1), nested.b(u+1));
dg::blas1::axpby( 1., nested.b(u+1), 1., nested.r(u+1), nested.b(u+1));
dg::blas1::copy( nested.x(u+1), nested.w(u+1)); 
}

unsigned s = nested.stages()-1;
try{
dg::apply( inverse_ops_up[s], nested.b(s), nested.x(s));
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR on stage "<<s<<" of full multigrid");
throw;
}
dg::blas1::axpby( 1., nested.x(s), -1., nested.w(s), nested.x(s) );
dg::blas2::symv( 1., nested.interpolation(s-1), nested.x(s), 1.,
nested.x(s-1));

for( int p=nested.stages()-2; p>=1; p--)
{
for( unsigned u=0; u<mu; u++)
multigrid_cycle( ops, inverse_ops_down, inverse_ops_up, nested, gamma, p);
dg::blas1::axpby( 1., nested.x(p), -1., nested.w(p), nested.x(p) );
dg::blas2::symv( 1., nested.interpolation(p-1), nested.x(p), 1.,
nested.x(p-1));
}
dg::blas1::copy( b, nested.b(0));
for( unsigned u=0; u<mu; u++)
multigrid_cycle( ops, inverse_ops_down, inverse_ops_up, nested, gamma, 0);
dg::blas1::copy( nested.x(0), x);
}


template<class NestedGrids, class MatrixType0, class MatrixType1, class MatrixType2,
class ContainerType0, class ContainerType1, class ContainerType2>
void fmg_solve(
std::vector<MatrixType0>& ops,
ContainerType0& x, const ContainerType1& b,
std::vector<MatrixType1>& inverse_ops_down, 
std::vector<MatrixType2>& inverse_ops_up, 
NestedGrids& nested_grids,
const ContainerType2& weights, double eps, unsigned gamma)
{
double nrmb = sqrt( blas2::dot( weights, b));

try{
full_multigrid( ops, x, b, inverse_ops_down, inverse_ops_up, nested_grids, gamma, 1);
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR in fmg_solve");
throw;
}

dg::apply( ops[0], x, nested_grids.r(0));
dg::blas1::axpby( 1., b, -1., nested_grids.r(0));
double error = sqrt( blas2::dot(weights,nested_grids.r(0)) );

while ( error >  eps*(nrmb + 1))
{
try{
full_multigrid( ops, x, b, inverse_ops_down, inverse_ops_up, nested_grids, gamma, 1);
}catch( dg::Error& err){
err.append_line( dg::Message(_ping_)<<"ERROR in fmg_solve");
throw;
}

blas2::symv( ops[0], x, nested_grids.r(0));
dg::blas1::axpby( 1., b, -1., nested_grids.r(0));
error = sqrt( blas2::dot(weights,nested_grids.r(0)) );
}
}




template< class Geometry, class Matrix, class Container>
struct MultigridCG2d
{
using geometry_type = Geometry;
using matrix_type = Matrix;
using container_type = Container;
using value_type = get_value_type<Container>;
MultigridCG2d() = default;

template<class ...ContainerParams>
MultigridCG2d( const Geometry& grid, const unsigned stages,
ContainerParams&& ... ps):
m_nested( grid, stages, std::forward<ContainerParams>(ps)...),
m_pcg(    stages), m_stages(stages)
{
for (unsigned u = 0; u < stages; u++)
m_pcg[u].construct(m_nested.x(u), m_nested.grid(u).size());
}


template<class ...Params>
void construct( Params&& ...ps)
{
*this = MultigridCG2d( std::forward<Params>( ps)...);
}

template<class ContainerType0>
void project( const ContainerType0& src, std::vector<ContainerType0>& out) const
{
m_nested.project( src, out);
}

template<class ContainerType0>
std::vector<ContainerType0> project( const ContainerType0& src) const
{
return m_nested.project( src);
}
unsigned stages()const{return m_nested.stages();}
unsigned num_stages()const{return m_nested.num_stages();}

const Geometry& grid( unsigned stage) const {
return m_nested.grid(stage);
}


unsigned max_iter() const{return m_pcg[0].get_max();}

void set_max_iter(unsigned new_max){ m_pcg[0].set_max(new_max);}

void set_benchmark( bool benchmark, std::string message = "Nested Iterations"){
m_benchmark = benchmark;
m_message = message;
}

const Container& copyable() const {return m_nested.copyable();}

template<class MatrixType, class ContainerType0, class ContainerType1>
std::vector<unsigned> solve( std::vector<MatrixType>& ops, ContainerType0&  x, const ContainerType1& b, value_type eps)
{
std::vector<value_type> v_eps( m_stages, eps);
for( unsigned u=m_stages-1; u>0; u--)
v_eps[u] = eps;
return solve( ops, x, b, v_eps);
}
template<class MatrixType, class ContainerType0, class ContainerType1>
std::vector<unsigned> solve( std::vector<MatrixType>& ops, ContainerType0&  x, const ContainerType1& b, std::vector<value_type> eps)
{
#ifdef MPI_VERSION
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 
std::vector<unsigned> number(m_stages);
std::vector<std::function<void( const ContainerType1&, ContainerType0&)> >
multi_inv_pol(m_stages);
for(unsigned u=0; u<m_stages; u++)
{
multi_inv_pol[u] = [&, u, &pcg = m_pcg[u], &pol = ops[u]](
const auto& y, auto& x)
{
dg::Timer t;
t.tic();
if ( u == 0)
number[u] = pcg.solve( pol, x, y, pol.precond(),
pol.weights(), eps[u], 1, 1);
else
number[u] = pcg.solve( pol, x, y, pol.precond(),
pol.weights(), eps[u], 1, 10);
t.toc();
if( m_benchmark)
DG_RANK0 std::cout << "# `"<<m_message<<"` stage: " << u << ", iter: " << number[u] << ", took "<<t.diff()<<"s\n";
};
}
nested_iterations( ops, x, b, multi_inv_pol, m_nested);

return number;
}

private:
dg::NestedGrids<Geometry, Matrix, Container> m_nested;
std::vector< PCG<Container> > m_pcg;
unsigned m_stages;
bool m_benchmark = true;
std::string m_message = "Nested Iterations";

};

}
