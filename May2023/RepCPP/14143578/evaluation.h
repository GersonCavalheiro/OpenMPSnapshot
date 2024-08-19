#pragma once

#include <cassert>
#include <cmath>
#include <thrust/host_vector.h>
#include "dg/backend/config.h"
#include "grid.h"
#include "operator.h"


namespace dg
{
namespace create
{

template<class real_type>
thrust::host_vector<real_type> abscissas( const RealGrid1d<real_type>& g)
{
thrust::host_vector<real_type> abs(g.size());
for( unsigned i=0; i<g.N(); i++)
for( unsigned j=0; j<g.n(); j++)
{
real_type xmiddle = DG_FMA( g.h(), (real_type)(i), g.x0());
real_type h2 = g.h()/2.;
real_type absj = 1.+g.dlt().abscissas()[j];
abs[i*g.n()+j] = DG_FMA( h2, absj, xmiddle);
}
return abs;
}
}




template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const RealGrid1d<real_type>& g)
{
thrust::host_vector<real_type> abs = create::abscissas( g);
for( unsigned i=0; i<g.size(); i++)
abs[i] = f( abs[i]);
return abs;
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const RealGrid1d<real_type>& g)
{
thrust::host_vector<real_type> v = evaluate<real_type (real_type)>( *f, g);
return v;
};



template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aRealTopology2d<real_type>& g)
{
thrust::host_vector<real_type> absx = create::abscissas( g.gx());
thrust::host_vector<real_type> absy = create::abscissas( g.gy());

thrust::host_vector<real_type> v( g.size());
for( unsigned i=0; i<g.Ny(); i++)
for( unsigned k=0; k<g.ny(); k++)
for( unsigned j=0; j<g.Nx(); j++)
for( unsigned r=0; r<g.nx(); r++)
v[ ((i*g.ny()+k)*g.Nx() + j)*g.nx() + r] =
f( absx[j*g.nx()+r], absy[i*g.ny()+k]);
return v;
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aRealTopology2d<real_type>& g)
{
return evaluate<real_type(real_type, real_type)>( *f, g);
};


template< class TernaryOp,class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aRealTopology3d<real_type>& g)
{
thrust::host_vector<real_type> absx = create::abscissas( g.gx());
thrust::host_vector<real_type> absy = create::abscissas( g.gy());
thrust::host_vector<real_type> absz = create::abscissas( g.gz());

thrust::host_vector<real_type> v( g.size());
for( unsigned s=0; s<g.Nz(); s++)
for( unsigned ss=0; ss<g.nz(); ss++)
for( unsigned i=0; i<g.Ny(); i++)
for( unsigned ii=0; ii<g.ny(); ii++)
for( unsigned k=0; k<g.Nx(); k++)
for( unsigned kk=0; kk<g.nx(); kk++)
v[ ((((s*g.nz()+ss)*g.Ny()+i)*g.ny()+ii)*g.Nx() + k)*g.nx() + kk] =
f( absx[k*g.nx()+kk], absy[i*g.ny()+ii], absz[s*g.nz()+ss]);
return v;
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aRealTopology3d<real_type>& g)
{
return evaluate<real_type(real_type, real_type, real_type)>( *f, g);
};


template<class real_type>
thrust::host_vector<real_type> integrate( const thrust::host_vector<real_type>& in, const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
double h = g.h();
unsigned n = g.n();
thrust::host_vector<real_type> to_out(g.size(), 0.);
thrust::host_vector<real_type> to_in(in);
if( dir == dg::backward ) 
{
for( unsigned i=0; i<in.size(); i++)
to_in[i] = in[ in.size()-1-i];
}


dg::Operator<real_type> forward = g.dlt().forward();
dg::Operator<real_type> backward = g.dlt().backward();
dg::Operator<real_type> ninj = create::ninj<real_type>( n );
Operator<real_type> t = create::pipj_inv<real_type>(n);
t *= h/2.;
ninj = backward*t*ninj*forward;
real_type constant = 0.;

for( unsigned i=0; i<g.N(); i++)
{
for( unsigned k=0; k<n; k++)
{
for( unsigned l=0; l<n; l++)
to_out[ i*n + k] += ninj(k,l)*to_in[ i*n + l];
to_out[ i*n + k] += constant;
}
for( unsigned l=0; l<n; l++)
constant += h*forward(0,l)*to_in[i*n+l];
}
thrust::host_vector<real_type> out(to_out);
if( dir == dg::backward ) 
{
for( unsigned i=0; i<in.size(); i++)
out[i] = -to_out[ in.size()-1-i]; 
}
return out;
}



template< class UnaryOp,class real_type>
thrust::host_vector<real_type> integrate( UnaryOp f, const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
thrust::host_vector<real_type> vector = evaluate( f, g);
return integrate<real_type>(vector, g, dir);
}
template<class real_type>
thrust::host_vector<real_type> integrate( real_type (f)(real_type), const RealGrid1d<real_type>& g, dg::direction dir = dg::forward)
{
thrust::host_vector<real_type> vector = evaluate( f, g);
return integrate<real_type>(vector, g, dir);
};

}

