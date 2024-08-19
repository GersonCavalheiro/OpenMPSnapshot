#pragma once

#include <cusp/coo_matrix.h>
#include "xspacelib.h"
#ifdef MPI_VERSION
#include "mpi_projection.h" 
#endif 


namespace dg
{
namespace create
{
namespace detail
{
template<class real_type>
void set_boundary(
cusp::array1d<real_type, cusp::host_memory>& values,
cusp::array1d<int, cusp::host_memory>& column_indices,
dg::bc bcx,
int num_cols)
{
for( unsigned k=0; k<values.size(); k++)
{
if( column_indices[k] < 0 )
{
if( bcx == dg::NEU || bcx == dg::NEU_DIR)
column_indices[k] = -(column_indices[k]+1);
else if( bcx == dg::DIR || bcx == dg::DIR_NEU)
{
column_indices[k] = -(column_indices[k]+1);
values[k] *= -1;
}
else if( bcx == dg::PER)
column_indices[k] += num_cols;
}
else if( column_indices[k] >= num_cols)
{
if( bcx == dg::NEU || bcx == dg::DIR_NEU)
column_indices[k] = 2*num_cols-1-column_indices[k];
else if( bcx == dg::DIR || bcx == dg::NEU_DIR)
{
column_indices[k] = 2*num_cols-1-column_indices[k];
values[k] *= -1;
}
else if( bcx == dg::PER)
column_indices[k] -= num_cols;
}
}
}

template<class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> window_stencil(
unsigned stencil_size,
const RealGrid1d<real_type>& local,
const RealGrid1d<real_type>& global,
dg::bc bcx)
{
cusp::array1d<real_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_offsets;
cusp::array1d<int, cusp::host_memory> column_indices;

unsigned num_rows = local.size();
unsigned num_cols = global.size();
unsigned radius = stencil_size/2;
int L0 = round((local.x0() - global.x0())/global.h())*global.n();

row_offsets.push_back(0);
for( unsigned k=0; k<num_rows; k++)
{
row_offsets.push_back(stencil_size + row_offsets[k]);
for( unsigned l=0; l<stencil_size; l++)
{
column_indices.push_back( L0 + (int)(k + l) - (int)radius);
values.push_back( 1.0);
}
}
set_boundary( values, column_indices, bcx, num_cols);

cusp::csr_matrix<int, real_type, cusp::host_memory> A(
num_rows, num_cols, values.size());
A.row_offsets = row_offsets;
A.column_indices = column_indices;
A.values = values;
return A;
}

template<class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> limiter_stencil(
const RealGrid1d<real_type>& local,
const RealGrid1d<real_type>& global,
dg::bc bcx)
{
cusp::array1d<real_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_offsets;
cusp::array1d<int, cusp::host_memory> column_indices;

unsigned num_rows = local.size();
unsigned num_cols = global.size();
int L0 = round((local.x0() - global.x0())/global.h())*global.n();
dg::Operator<real_type> forward = local.dlt().forward();
dg::Operator<real_type> backward = local.dlt().backward();
if( global.n() == 1)
throw dg::Error( dg::Message(_ping_) << "Limiter stencil not possible for n==1!");


row_offsets.push_back( 0);
for( unsigned k=0; k<local.N(); k++)
{
row_offsets.push_back(row_offsets[row_offsets.size()-1] + 3*local.n());
for( unsigned j=1; j<local.n(); j++)
row_offsets.push_back(row_offsets[row_offsets.size()-1]);
for( unsigned j=0; j<local.n(); j++)
{
column_indices.push_back( L0 + (int)((k-1)*global.n() + j) );
values.push_back( forward(0, j ));
}
for( unsigned j=0; j<local.n(); j++)
{
column_indices.push_back( L0 + (int)(k*global.n() + j ));
values.push_back( forward(1, j ));
}
for( unsigned j=0; j<local.n(); j++)
{
column_indices.push_back( L0 + (int)((k+1)*global.n() + j));
values.push_back( backward(j, 1) );
}
}
assert( row_offsets.size() == num_rows+1);
set_boundary( values, column_indices, bcx, num_cols);

cusp::csr_matrix<int, real_type, cusp::host_memory> A(
num_rows, num_cols, values.size());
A.row_offsets = row_offsets;
A.column_indices = column_indices;
A.values = values;
return A;
}

template<class real_type>
cusp::csr_matrix< int, real_type, cusp::host_memory> identity_matrix( const RealGrid1d<real_type>& local, const RealGrid1d<real_type>& global)
{
cusp::csr_matrix<int,real_type,cusp::host_memory> A( local.size(),
global.size(), local.size());
int L0 = round((local.x0() - global.x0())/global.h())*global.n();
A.row_offsets[0] = 0;
for( unsigned i=0; i<local.size(); i++)
{
A.row_offsets[i+1] = 1 + A.row_offsets[i];
A.column_indices[i] = L0 + i;
A.values[i] = 1.;
}
return A;
}

} 



template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
unsigned window_size,
const RealGrid1d<real_type>& g,
dg::bc bcx)
{
return detail::window_stencil( window_size, g, g, bcx);
}


template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
const RealGrid1d<real_type>& g,
dg::bc bound)
{
return detail::limiter_stencil( g, g, bound);
}




template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
std::array<int,2> window_size,
const aRealTopology2d<real_type>& g,
dg::bc bcx, dg::bc bcy)
{
auto mx = detail::window_stencil(window_size[0], g.gx(), g.gx(), bcx);
auto my = detail::window_stencil(window_size[1], g.gy(), g.gy(), bcy);
return dg::tensorproduct( my, mx);
}

template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
enum coo3d direction,
const aRealTopology2d<real_type>& g,
dg::bc bound)
{
if( direction == dg::coo3d::x)
{
auto mx = detail::limiter_stencil(g.gx(), g.gx(), bound);
auto einsy = detail::identity_matrix( g.gy(), g.gy());
return dg::tensorproduct( einsy, mx);
}
auto my = detail::limiter_stencil(g.gy(), g.gy(), bound);
auto einsx = detail::identity_matrix( g.gx(), g.gx());
return dg::tensorproduct( my, einsx);
}

template<class real_type>
dg::IHMatrix_t<real_type> limiter_stencil(
enum coo3d direction,
const aRealTopology3d<real_type>& g,
dg::bc bound)
{
if( direction == dg::coo3d::x)
{
auto mx = detail::limiter_stencil(g.gx(), g.gx(), bound);
auto einsy = detail::identity_matrix( g.gy(), g.gy());
auto einsz = detail::identity_matrix( g.gz(), g.gz());
auto temp = dg::tensorproduct( einsy, mx);
return dg::tensorproduct( einsz, temp);
}
if( direction == dg::coo3d::y)
{
auto einsx = detail::identity_matrix( g.gx(), g.gx());
auto my = detail::limiter_stencil(g.gy(), g.gy(), bound);
auto einsz = detail::identity_matrix( g.gz(), g.gz());
return dg::tensorproduct( einsz, dg::tensorproduct( my, einsx));
}
auto mz = detail::limiter_stencil(g.gz(), g.gz(), bound);
auto einsy = detail::identity_matrix( g.gy(), g.gy());
auto einsx = detail::identity_matrix( g.gx(), g.gx());
return dg::tensorproduct( mz, dg::tensorproduct( einsy, einsx));
}


template<class real_type>
dg::IHMatrix_t<real_type> window_stencil(
std::array<int,2> window_size,
const aRealTopology3d<real_type>& g,
dg::bc bcx, dg::bc bcy)
{
auto mx = detail::window_stencil(window_size[0], g.gx(), g.gx(), bcx);
auto my = detail::window_stencil(window_size[1], g.gy(), g.gy(), bcy);
auto mz = detail::identity_matrix( g.gz(), g.gz());
return dg::tensorproduct( mz, dg::tensorproduct( my, mx));
}

#ifdef MPI_VERSION
template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
std::array<int,2> window_size,
const aRealMPITopology2d<real_type>& g,
dg::bc bcx, dg::bc bcy)
{
auto mx = detail::window_stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
auto my = detail::window_stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
auto local = dg::tensorproduct( my, mx);
return dg::convert( local, g);
}

template<class real_type>
dg::MIHMatrix_t<real_type> window_stencil(
std::array<int,2> window_size,
const aRealMPITopology3d<real_type>& g,
dg::bc bcx, dg::bc bcy)
{
auto mx = detail::window_stencil(window_size[0], g.local().gx(), g.global().gx(), bcx);
auto my = detail::window_stencil(window_size[1], g.local().gy(), g.global().gy(), bcy);
auto mz = detail::identity_matrix( g.local().gz(), g.global().gz());
auto out = dg::tensorproduct( mz, dg::tensorproduct( my, mx));
return dg::convert( out, g);
}

template<class real_type>
dg::MIHMatrix_t<real_type> limiter_stencil(
enum coo3d direction,
const aRealMPITopology2d<real_type>& g,
dg::bc bound)
{
if( direction == dg::coo3d::x)
{
auto mx = detail::limiter_stencil(g.local().gx(), g.global().gx(), bound);
auto einsy = detail::identity_matrix( g.local().gy(), g.global().gy());
auto local = dg::tensorproduct( einsy, mx);
return dg::convert( (dg::IHMatrix)local, g);
}
auto my = detail::limiter_stencil(g.local().gy(), g.global().gy(), bound);
auto einsx = detail::identity_matrix( g.local().gx(), g.global().gx());
auto local = dg::tensorproduct( my, einsx);
return dg::convert( local, g);
}

template<class real_type>
dg::MIHMatrix_t<real_type> limiter_stencil(
enum coo3d direction,
const aRealMPITopology3d<real_type>& g,
dg::bc bound)
{
if( direction == dg::coo3d::x)
{
auto mx = detail::limiter_stencil(g.local().gx(), g.global().gx(), bound);
auto einsy = detail::identity_matrix( g.local().gy(), g.global().gy());
auto einsz = detail::identity_matrix( g.local().gz(), g.global().gz());
auto local = dg::tensorproduct( einsz, dg::tensorproduct( einsy, mx));
return dg::convert( local, g);
}
if( direction == dg::coo3d::y)
{
auto einsx = detail::identity_matrix( g.local().gx(), g.global().gx());
auto my = detail::limiter_stencil(g.local().gy(), g.global().gy(), bound);
auto einsz = detail::identity_matrix( g.local().gz(), g.global().gz());
auto local = dg::tensorproduct( einsz, dg::tensorproduct( my, einsx));
return dg::convert( local, g);
}
auto mz = detail::limiter_stencil(g.local().gz(), g.global().gz(), bound);
auto einsy = detail::identity_matrix( g.local().gy(), g.global().gy());
auto einsx = detail::identity_matrix( g.local().gx(), g.global().gx());
auto local = dg::tensorproduct( mz, dg::tensorproduct( einsy, einsx));
return dg::convert( local, g);
}

#endif 

} 
} 
