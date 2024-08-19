#pragma once
#include "weights.h"


namespace dg {
namespace create{
namespace detail
{
template<class real_type>
std::vector<real_type> fem_weights( const DLT<real_type>& dlt)
{
std::vector<real_type> x = dlt.abscissas();
std::vector<real_type> w = x;
unsigned n = x.size();
if( n== 1)
w[0] = 2;
else
{
w[0] = 0.5*(x[1] - (x[n-1]-2));
w[n-1] = 0.5*((x[0]+2) - x[n-2]);
for( unsigned i=1; i<n-1; i++)
w[i] = 0.5*(x[i+1]-x[i-1]);
}
return w;
}
}





template<class real_type>
thrust::host_vector<real_type> fem_weights( const RealGrid1d<real_type>& g)
{
thrust::host_vector<real_type> v( g.size());
std::vector<real_type> w = detail::fem_weights(g.dlt());
for( unsigned i=0; i<g.N(); i++)
for( unsigned j=0; j<g.n(); j++)
v[i*g.n() + j] = g.h()/2.*w[j];
return v;
}
template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const RealGrid1d<real_type>& g)
{
thrust::host_vector<real_type> v = fem_weights( g);
for( unsigned i=0; i<g.size(); i++)
v[i] = 1./v[i];
return v;
}

template<class real_type>
thrust::host_vector<real_type> fem_weights( const aRealTopology2d<real_type>& g)
{
thrust::host_vector<real_type> v( g.size());
std::vector<real_type> wx = detail::fem_weights(g.dltx());
std::vector<real_type> wy = detail::fem_weights(g.dlty());
for( unsigned i=0; i<g.size(); i++)
v[i] = g.hx()*g.hy()/4.*
wx[i%g.nx()]*
wy[(i/(g.nx()*g.Nx()))%g.ny()];
return v;
}
template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const aRealTopology2d<real_type>& g)
{
thrust::host_vector<real_type> v = fem_weights( g);
for( unsigned i=0; i<g.size(); i++)
v[i] = 1./v[i];
return v;
}

template<class real_type>
thrust::host_vector<real_type> fem_weights( const aRealTopology3d<real_type>& g)
{
thrust::host_vector<real_type> v( g.size());
std::vector<real_type> wx = detail::fem_weights(g.dltx());
std::vector<real_type> wy = detail::fem_weights(g.dlty());
std::vector<real_type> wz = detail::fem_weights(g.dltz());
for( unsigned i=0; i<g.size(); i++)
v[i] = g.hx()*g.hy()*g.hz()/8.*
wx[i%g.nx()]*
wy[(i/(g.nx()*g.Nx()))%g.ny()]*
wz[(i/(g.nx()*g.ny()*g.Nx()*g.Ny()))%g.nz()];
return v;
}

template<class real_type>
thrust::host_vector<real_type> fem_inv_weights( const aRealTopology3d<real_type>& g)
{
thrust::host_vector<real_type> v = fem_weights( g);
for( unsigned i=0; i<g.size(); i++)
v[i] = 1./v[i];
return v;
}
}
}
