#pragma once
#include <vector>
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include "grid.h"
#include "interpolation.h"
#include "weights.h"
#include "fem.h"


namespace dg{


template<class T>
T gcd( T a, T b)
{
T r2 = std::max(a,b);
T r1 = std::min(a,b);
while( r1!=0)
{
r2 = r2%r1;
std::swap( r1, r2);
}
return r2;
}


template<class T>
T lcm( T a, T b)
{
T g = gcd( a,b);
return a/g*b;
}

namespace create{


template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> diagonal( const thrust::host_vector<real_type>& diagonal)
{
unsigned size = diagonal.size();
cusp::coo_matrix<int, real_type, cusp::host_memory> W( size, size, size);
for( unsigned i=0; i<size; i++)
{
W.row_indices[i] = W.column_indices[i] = i;
W.values[i] = diagonal[i];
}
return W;
}



template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old, std::string method = "dg")
{
if( g_old.N() % g_new.N() != 0) std::cerr << "# ATTENTION: you project between incompatible grids!! old N: "<<g_old.N()<<" new N: "<<g_new.N()<<"\n";
if( g_old.n() < g_new.n()) std::cerr << "# ATTENTION: you project between incompatible grids!! old n: "<<g_old.n()<<" new n: "<<g_new.n()<<"\n";
cusp::coo_matrix<int, real_type, cusp::host_memory> Wf =
dg::create::diagonal( dg::create::weights( g_old));
cusp::coo_matrix<int, real_type, cusp::host_memory> Vc =
dg::create::diagonal( dg::create::inv_weights( g_new));
cusp::coo_matrix<int, real_type, cusp::host_memory> temp = interpolation( g_old, g_new, method), A;
cusp::transpose( temp, A);
cusp::multiply( A, Wf, temp);
cusp::multiply( Vc, temp, A);
A.sort_by_row_and_column();
return A;
}


template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old, std::string method = "dg")
{
cusp::csr_matrix<int, real_type, cusp::host_memory> projectX = projection( g_new.gx(), g_old.gx(), method);
cusp::csr_matrix<int, real_type, cusp::host_memory> projectY = projection( g_new.gy(), g_old.gy(), method);
return dg::tensorproduct( projectY, projectX);
}

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old, std::string method = "dg")
{
cusp::csr_matrix<int, real_type, cusp::host_memory> projectX = projection( g_new.gx(), g_old.gx(), method);
cusp::csr_matrix<int, real_type, cusp::host_memory> projectY = projection( g_new.gy(), g_old.gy(), method);
cusp::csr_matrix<int, real_type, cusp::host_memory> projectZ = projection( g_new.gz(), g_old.gz(), method);
return dg::tensorproduct( projectZ, dg::tensorproduct( projectY, projectX));
}


template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old)
{
Grid3d g_lcm( Grid1d( g_new.x0(), g_new.x1(), lcm( g_new.nx(), g_old.nx()), lcm( g_new.Nx(), g_old.Nx())),
Grid1d( g_new.y0(), g_new.y1(), lcm( g_new.ny(), g_old.ny()), lcm( g_new.Ny(), g_old.Ny())),
Grid1d( g_new.z0(), g_new.z1(), lcm( g_new.nz(), g_old.nz()), lcm( g_new.Nz(), g_old.Nz())));
cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
cusp::multiply( P, Q, Y);
Y.sort_by_row_and_column();
return Y;
}

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old)
{
Grid2d g_lcm( Grid1d( g_new.x0(), g_new.x1(), lcm( g_new.nx(), g_old.nx()), lcm( g_new.Nx(), g_old.Nx())),
Grid1d( g_new.y0(), g_new.y1(), lcm( g_new.ny(), g_old.ny()), lcm( g_new.Ny(), g_old.Ny())));
cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
cusp::multiply( P, Q, Y);
Y.sort_by_row_and_column();
return Y;
}
template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> transformation( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old)
{
RealGrid1d<real_type> g_lcm(g_new.x0(), g_new.x1(), lcm(g_new.n(), g_old.n()), lcm(g_new.N(), g_old.N()));
cusp::coo_matrix< int, real_type, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
cusp::coo_matrix< int, real_type, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
cusp::multiply( P, Q, Y);
Y.sort_by_row_and_column();
return Y;
}


template<class real_type>
dg::IHMatrix_t<real_type> backproject( const RealGrid1d<real_type>& g)
{
unsigned n=g.n();
dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
auto block = dg::create::transformation( g_new, g_old);
dg::Operator<real_type> op(n, 0.);
for( unsigned i=0; i<block.num_entries; i++)
op( block.row_indices[i], block.column_indices[i]) = block.values[i];

return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), op);
}

template<class real_type>
dg::IHMatrix_t<real_type> backproject( const aRealTopology2d<real_type>& g)
{
auto transformX = backproject( g.gx());
auto transformY = backproject( g.gy());
return dg::tensorproduct( transformY, transformX);
}

template<class real_type>
dg::IHMatrix_t<real_type> backproject( const aRealTopology3d<real_type>& g)
{
auto transformX = backproject( g.gx());
auto transformY = backproject( g.gy());
auto transformZ = backproject( g.gz());
return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
}


template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const RealGrid1d<real_type>& g)
{
unsigned n=g.n();
dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
dg::RealGrid1d<real_type> g_new( -1., 1., 1, n);
auto block = dg::create::transformation( g_new, g_old);
dg::Operator<real_type> op(n, 0.);
for( unsigned i=0; i<block.num_entries; i++)
op( block.row_indices[i], block.column_indices[i]) = block.values[i];

return (dg::IHMatrix_t<real_type>)dg::tensorproduct( g.N(), dg::invert(op));
}

template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const aRealTopology2d<real_type>& g)
{
auto transformX = inv_backproject( g.gx());
auto transformY = inv_backproject( g.gy());
return dg::tensorproduct( transformY, transformX);
}

template<class real_type>
dg::IHMatrix_t<real_type> inv_backproject( const aRealTopology3d<real_type>& g)
{
auto transformX = inv_backproject( g.gx());
auto transformY = inv_backproject( g.gy());
auto transformZ = inv_backproject( g.gz());
return dg::tensorproduct( transformZ, dg::tensorproduct(transformY, transformX));
}


}
}
