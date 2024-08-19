#pragma once

#include "dg/backend/mpi_vector.h"
#include "mpi_grid.h"
#include "evaluation.h"


namespace dg
{




template< class BinaryOp,class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( const BinaryOp& f, const aRealMPITopology2d<real_type>& g)
{
RealGrid2d<real_type> l = g.local();
int dims[2], periods[2], coords[2];
MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
thrust::host_vector<real_type> absx( l.nx()*l.Nx());
thrust::host_vector<real_type> absy( l.ny()*l.Ny());
for( unsigned i=0; i<l.Nx(); i++)
for( unsigned j=0; j<l.nx(); j++)
{
unsigned coord = i+l.Nx()*coords[0];
real_type xmiddle = DG_FMA( g.hx(), (real_type)(coord), g.x0());
real_type h2 = g.hx()/2.;
real_type absj = 1.+g.dltx().abscissas()[j];
absx[i*l.nx()+j] = DG_FMA( h2, absj, xmiddle);
}
for( unsigned i=0; i<l.Ny(); i++)
for( unsigned j=0; j<l.ny(); j++)
{
unsigned coord = i+l.Ny()*coords[1];
real_type ymiddle = DG_FMA( g.hy(), (real_type)(coord), g.y0());
real_type h2 = g.hy()/2.;
real_type absj = 1.+g.dlty().abscissas()[j];
absy[i*l.ny()+j] = DG_FMA( h2, absj, ymiddle );
}

thrust::host_vector<real_type> w( l.size());
for( unsigned i=0; i<l.Ny(); i++)
for( unsigned k=0; k<l.ny(); k++)
for( unsigned j=0; j<l.Nx(); j++)
for( unsigned r=0; r<l.nx(); r++)
w[ ((i*l.ny()+k)*l.Nx() + j)*l.nx() + r] = f( absx[j*l.nx()+r], absy[i*l.ny()+k]);
MPI_Vector<thrust::host_vector<real_type> > v( w, g.communicator());
return v;
};
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( real_type(f)(real_type, real_type), const aRealMPITopology2d<real_type>& g)
{
return evaluate<real_type(real_type, real_type)>( *f, g);
};


template< class TernaryOp,class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( const TernaryOp& f, const aRealMPITopology3d<real_type>& g)
{
RealGrid3d<real_type> l = g.local();
int dims[3], periods[3], coords[3];
MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
thrust::host_vector<real_type> absx( l.nx()*l.Nx());
thrust::host_vector<real_type> absy( l.ny()*l.Ny());
thrust::host_vector<real_type> absz( l.nz()*l.Nz());
for( unsigned i=0; i<l.Nx(); i++)
for( unsigned j=0; j<l.nx(); j++)
{
unsigned coord = i+l.Nx()*coords[0];
real_type xmiddle = DG_FMA( g.hx(), (real_type)(coord), g.x0());
real_type h2 = g.hx()/2.;
real_type absj = 1.+g.dltx().abscissas()[j];
absx[i*l.nx()+j] = DG_FMA( h2, absj, xmiddle);
}
for( unsigned i=0; i<l.Ny(); i++)
for( unsigned j=0; j<l.ny(); j++)
{
unsigned coord = i+l.Ny()*coords[1];
real_type ymiddle = DG_FMA( g.hy(), (real_type)(coord), g.y0());
real_type h2 = g.hy()/2.;
real_type absj = 1.+g.dlty().abscissas()[j];
absy[i*l.ny()+j] = DG_FMA( h2, absj, ymiddle );
}
for( unsigned i=0; i<l.Nz(); i++)
for( unsigned j=0; j<l.nz(); j++)
{
unsigned coord = i+l.Nz()*coords[2];
real_type zmiddle = DG_FMA( g.hz(), (real_type)(coord), g.z0());
real_type h2 = g.hz()/2.;
real_type absj = 1.+g.dltz().abscissas()[j];
absz[i*l.nz()+j] = DG_FMA( h2, absj, zmiddle );
}

thrust::host_vector<real_type> w( l.size());
for( unsigned s=0; s<l.Nz(); s++)
for( unsigned m=0; m<l.nz(); m++)
for( unsigned i=0; i<l.Ny(); i++)
for( unsigned k=0; k<l.ny(); k++)
for( unsigned j=0; j<l.Nx(); j++)
for( unsigned r=0; r<l.nx(); r++)
w[ ((((s*l.nz()+m)*l.Ny()+i)*l.ny()+k)*l.Nx() + j)*l.nx() + r] =
f( absx[j*l.nx()+r], absy[i*l.ny()+k], absz[s*l.nz()+m]);
MPI_Vector<thrust::host_vector<real_type> > v( w, g.communicator());
return v;
};
template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > evaluate( real_type(f)(real_type, real_type, real_type), const aRealMPITopology3d<real_type>& g)
{
return evaluate<real_type(real_type, real_type, real_type)>( *f, g);
};


template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > global2local( const thrust::host_vector<real_type>& global, const aRealMPITopology3d<real_type>& g)
{
assert( global.size() == g.global().size());
RealGrid3d<real_type> l = g.local();
thrust::host_vector<real_type> temp(l.size());
int dims[3], periods[3], coords[3];
MPI_Cart_get( g.communicator(), 3, dims, periods, coords);
for( unsigned s=0; s<l.nz()*l.Nz(); s++)
for( unsigned i=0; i<l.ny()*l.Ny(); i++)
for( unsigned j=0; j<l.nx()*l.Nx(); j++)
{
unsigned idx1 = (s*l.ny()*l.Ny()+i)*l.nx()*l.Nx() + j;
unsigned idx2 = ((((coords[2]*l.nz()*l.Nz()+s)*dims[1]
+coords[1])*l.ny()*l.Ny()+i)*dims[0]
+coords[0])*l.nx()*l.Nx() + j;
temp[idx1] = global[idx2];
}
return MPI_Vector<thrust::host_vector<real_type> >(temp, g.communicator());
}

template<class real_type>
MPI_Vector<thrust::host_vector<real_type> > global2local( const thrust::host_vector<real_type>& global, const aRealMPITopology2d<real_type>& g)
{
assert( global.size() == g.global().size());
RealGrid2d<real_type> l = g.local();
thrust::host_vector<real_type> temp(l.size());
int dims[2], periods[2], coords[2];
MPI_Cart_get( g.communicator(), 2, dims, periods, coords);
for( unsigned i=0; i<l.ny()*l.Ny(); i++)
for( unsigned j=0; j<l.nx()*l.Nx(); j++)
{
unsigned idx1 = i*l.nx()*l.Nx() + j;
unsigned idx2 = ((coords[1]*l.ny()*l.Ny()+i)*dims[0]
+ coords[0])*l.nx()*l.Nx() + j;
temp[idx1] = global[idx2];
}
return MPI_Vector<thrust::host_vector<real_type> >(temp, g.communicator());
}

}

