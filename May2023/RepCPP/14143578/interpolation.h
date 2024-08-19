#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "dg/backend/typedefs.h"
#include "grid.h"
#include "evaluation.h"
#include "functions.h"
#include "operator_tensor.h"
#include "xspacelib.h"



namespace dg{

namespace create{
namespace detail{


template<class real_type>
std::vector<real_type> coefficients( real_type xn, unsigned n)
{
assert( xn <= 1. && xn >= -1.);
std::vector<real_type> px(n);
if( xn == -1)
{
for( unsigned u=0; u<n; u++)
px[u] = (u%2 == 0) ? +1. : -1.;
}
else if( xn == 1)
{
for( unsigned i=0; i<n; i++)
px[i] = 1.;
}
else
{
px[0] = 1.;
if( n > 1)
{
px[1] = xn;
for( unsigned i=1; i<n-1; i++)
px[i+1] = ((real_type)(2*i+1)*xn*px[i]-(real_type)i*px[i-1])/(real_type)(i+1);
}
}
return px;
}

template<class real_type>
std::vector<real_type> lagrange( real_type x, const std::vector<real_type>& xi)
{
unsigned n = xi.size();
std::vector<real_type> l( n , 1.);
for( unsigned i=0; i<n; i++)
for( unsigned k=0; k<n; k++)
{
if ( k != i)
l[i] *= (x-xi[k])/(xi[i]-xi[k]);
}
return l;
}

template<class real_type>
std::vector<real_type> choose_1d_abscissas( real_type X,
unsigned points_per_line, const RealGrid1d<real_type>& g,
const thrust::host_vector<real_type>& abs,
thrust::host_vector<unsigned>& cols)
{
assert( abs.size() >= points_per_line && "There must be more points to interpolate\n");
dg::bc bcx = g.bcx();
real_type xnn = (X-g.x0())/g.h();
unsigned n = (unsigned)floor(xnn);
if (n==g.N() && bcx != dg::PER) {
n-=1;
}
std::vector<real_type> xs( points_per_line, 0);
auto it = std::lower_bound( abs.begin()+n*g.n(), abs.begin() + (n+1)*g.n(),
X);
cols.resize( points_per_line, 0);
switch( points_per_line)
{
case 1: xs[0] = 1.;
if( it == abs.begin())
cols[0] = 0;
else if( it == abs.end())
cols[0] = it - abs.begin() - 1;
else
{
if ( fabs(X - *it) < fabs( X - *(it-1)))
cols[0] = it - abs.begin();
else
cols[0] = it - abs.begin() -1;
}
break;
case 2: if( it == abs.begin())
{
if( bcx == dg::PER)
{
xs[0] = *it;
xs[1] = *(abs.end() -1)-g.lx();;
cols[0] = 0, cols[1] = abs.end()-abs.begin()-1;
}
else
{
xs.resize(1);
xs[0] = *it;
cols[0] = 0;
}
}
else if( it == abs.end())
{
if( bcx == dg::PER)
{
xs[0] = *(abs.begin())+g.lx();
xs[1] = *(it-1);
cols[0] = 0, cols[1] = it-abs.begin()-1;
}
else
{
xs.resize(1);
xs[0] = *(it-1);
cols[0] = it-abs.begin()-1;
}
}
else
{
xs[0] = *(it-1);
xs[1] = *it;
cols[0] = it - abs.begin() - 1;
cols[1] = cols[0]+1;
}
break;
case 4: if( it <= abs.begin() +1)
{
if( bcx == dg::PER)
{
xs[0] = *abs.begin(), cols[0] = 0;
xs[1] = *(abs.begin()+1), cols[1] = 1;
xs[2] = it == abs.begin() ? *(abs.end() -2) : *(abs.begin()+2);
cols[2] = it == abs.begin() ? abs.end()-abs.begin() -2 : 2;
xs[3] = *(abs.end() -1);
cols[3] = abs.end()-abs.begin() -1;
}
else
{
it = abs.begin();
xs[0] = *it,     xs[1] = *(it+1);
xs[2] = *(it+2), xs[3] = *(it+3);
cols[0] = 0, cols[1] = 1;
cols[2] = 2, cols[3] = 3;
}
}
else if( it >= abs.end() -1)
{
if( bcx == dg::PER)
{
xs[0] = *abs.begin(), cols[0] = 0;
xs[1] = it == abs.end() ? *(abs.begin()+1) : *(abs.end() -3) ;
cols[1] = it == abs.end() ? 1 :  abs.end()-abs.begin()-3 ;
xs[2] = *(abs.end() - 2), cols[2] = abs.end()-abs.begin()-2;
xs[3] = *(abs.end() - 1), cols[3] = abs.end()-abs.begin()-1;
}
else
{
it = abs.end();
xs[0] = *(it-4), xs[1] = *(it-3);
xs[2] = *(it-2), xs[3] = *(it-1);
cols[0] = it - abs.begin() - 4;
cols[1] = cols[0]+1;
cols[2] = cols[1]+1;
cols[3] = cols[2]+1;
}
}
else
{
xs[0] = *(it-2), xs[1] = *(it-1);
xs[2] = *(it  ), xs[3] = *(it+1);
cols[0] = it - abs.begin() - 2;
cols[1] = cols[0]+1;
cols[2] = cols[1]+1;
cols[3] = cols[2]+1;
}
break;
}
return xs;
}

}




template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
const thrust::host_vector<real_type>& x,
const RealGrid1d<real_type>& g,
dg::bc bcx = dg::NEU,
std::string method = "dg")
{
cusp::array1d<real_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_indices;
cusp::array1d<int, cusp::host_memory> column_indices;
if( method == "dg")
{
dg::Operator<real_type> forward( g.dlt().forward());
for( unsigned i=0; i<x.size(); i++)
{
real_type X = x[i];
bool negative = false;
g.shift( negative, X, bcx);

real_type xnn = (X-g.x0())/g.h();
unsigned n = (unsigned)floor(xnn);
real_type xn = 2.*xnn - (real_type)(2*n+1);
if (n==g.N()) {
n-=1;
xn = 1.;
}
std::vector<real_type> px = detail::coefficients( xn, g.n());
std::vector<real_type> pxF(px.size(),0);
for( unsigned l=0; l<g.n(); l++)
for( unsigned k=0; k<g.n(); k++)
pxF[l]+= px[k]*forward(k,l);
unsigned cols = n*g.n();
for ( unsigned l=0; l<g.n(); l++)
{
row_indices.push_back(i);
column_indices.push_back( cols + l);
values.push_back(negative ? -pxF[l] : pxF[l]);
}
}
}
else
{
unsigned points_per_line = 1;
if( method == "nearest")
points_per_line = 1;
else if( method == "linear")
points_per_line = 2;
else if( method == "cubic")
points_per_line = 4;
else
throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
thrust::host_vector<real_type> abs = dg::create::abscissas( g);
dg::RealGrid1d<real_type> gx( g.x0(), g.x1(), g.n(), g.N(), bcx);
for( unsigned i=0; i<x.size(); i++)
{
real_type X = x[i];
bool negative = false;
g.shift( negative, X, bcx);

thrust::host_vector<unsigned> cols;
std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
points_per_line, gx, abs, cols);

std::vector<real_type> px = detail::lagrange( X, xs);
for ( unsigned l=0; l<px.size(); l++)
{
row_indices.push_back(i);
column_indices.push_back( cols[l]);
values.push_back(negative ? -px[l] : px[l]);
}
}
}
cusp::coo_matrix<int, real_type, cusp::host_memory> A(
x.size(), g.size(), values.size());
A.row_indices = row_indices;
A.column_indices = column_indices;
A.values = values;
return A;
}


template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
const thrust::host_vector<real_type>& x,
const thrust::host_vector<real_type>& y,
const aRealTopology2d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
std::string method = "dg")
{
assert( x.size() == y.size());
cusp::array1d<real_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_indices;
cusp::array1d<int, cusp::host_memory> column_indices;
if( method == "dg")
{
std::vector<real_type> gauss_nodesx = g.dltx().abscissas();
std::vector<real_type> gauss_nodesy = g.dlty().abscissas();
dg::Operator<real_type> forwardx( g.dltx().forward());
dg::Operator<real_type> forwardy( g.dlty().forward());


for( int i=0; i<(int)x.size(); i++)
{
real_type X = x[i], Y = y[i];
bool negative=false;
g.shift( negative,X,Y, bcx, bcy);

real_type xnn = (X-g.x0())/g.hx();
real_type ynn = (Y-g.y0())/g.hy();
unsigned nn = (unsigned)floor(xnn);
unsigned mm = (unsigned)floor(ynn);
real_type xn =  2.*xnn - (real_type)(2*nn+1);
real_type yn =  2.*ynn - (real_type)(2*mm+1);
if (nn==g.Nx()) {
nn-=1;
xn =1.;
}
if (mm==g.Ny()) {
mm-=1;
yn =1.;
}
int idxX =-1, idxY = -1;
for( unsigned k=0; k<g.nx(); k++)
{
if( fabs( xn - gauss_nodesx[k]) < 1e-14)
idxX = nn*g.nx() + k; 
}
for( unsigned k=0; k<g.ny(); k++)
{
if( fabs( yn - gauss_nodesy[k]) < 1e-14)
idxY = mm*g.ny() + k;  
}
if( idxX < 0 && idxY < 0 ) 
{
std::vector<real_type> px = detail::coefficients( xn, g.nx()),
py = detail::coefficients( yn, g.ny());
std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0);
for( unsigned l=0; l<g.nx(); l++)
for( unsigned k=0; k<g.nx(); k++)
pxF[l]+= px[k]*forwardx(k,l);
for( unsigned l=0; l<g.ny(); l++)
for( unsigned k=0; k<g.ny(); k++)
pyF[l]+= py[k]*forwardy(k,l);
for(unsigned k=0; k<g.ny(); k++)
for( unsigned l=0; l<g.nx(); l++)
{
row_indices.push_back( i);
column_indices.push_back( ((mm*g.ny()+k)*g.Nx()+nn)*g.nx() + l);
real_type pxy = pyF[k]*pxF[l];
if( !negative)
values.push_back(  pxy);
else
values.push_back( -pxy);
}
}
else if ( idxX < 0 && idxY >=0) 
{
std::vector<real_type> px = detail::coefficients( xn, g.nx());
std::vector<real_type> pxF(g.nx(),0);
for( unsigned l=0; l<g.nx(); l++)
for( unsigned k=0; k<g.nx(); k++)
pxF[l]+= px[k]*forwardx(k,l);
for( unsigned l=0; l<g.nx(); l++)
{
row_indices.push_back( i);
column_indices.push_back( ((idxY)*g.Nx() + nn)*g.nx() + l);
if( !negative)
values.push_back( pxF[l]);
else
values.push_back(-pxF[l]);

}
}
else if ( idxX >= 0 && idxY < 0) 
{
std::vector<real_type> py = detail::coefficients( yn, g.ny());
std::vector<real_type> pyF(g.ny(),0);
for( unsigned l=0; l<g.ny(); l++)
for( unsigned k=0; k<g.ny(); k++)
pyF[l]+= py[k]*forwardy(k,l);
for( unsigned k=0; k<g.ny(); k++)
{
row_indices.push_back(i);
column_indices.push_back((mm*g.ny()+k)*g.Nx()*g.nx() + idxX);
if( !negative)
values.push_back( pyF[k]);
else
values.push_back(-pyF[k]);

}
}
else 
{
row_indices.push_back(i);
column_indices.push_back(idxY*g.Nx()*g.nx() + idxX);
if( !negative)
values.push_back( 1.);
else
values.push_back(-1.);
}

}
}
else
{
unsigned points_per_line = 1;
if( method == "nearest")
points_per_line = 1;
else if( method == "linear")
points_per_line = 2;
else if( method == "cubic")
points_per_line = 4;
else
throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
RealGrid1d<real_type> gx(g.x0(), g.x1(), g.nx(), g.Nx(), bcx);
RealGrid1d<real_type> gy(g.y0(), g.y1(), g.ny(), g.Ny(), bcy);
thrust::host_vector<real_type> absX = dg::create::abscissas( gx);
thrust::host_vector<real_type> absY = dg::create::abscissas( gy);

for( unsigned i=0; i<x.size(); i++)
{
real_type X = x[i], Y = y[i];
bool negative = false;
g.shift( negative, X, Y, bcx, bcy);

thrust::host_vector<unsigned> colsX, colsY;
std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
points_per_line, gx, absX, colsX);
std::vector<real_type> ys  = detail::choose_1d_abscissas( Y,
points_per_line, gy, absY, colsY);

std::vector<real_type> pxy( points_per_line*points_per_line);
std::vector<real_type> px = detail::lagrange( X, xs),
py = detail::lagrange( Y, ys);
for(unsigned k=0; k<py.size(); k++)
for( unsigned l=0; l<px.size(); l++)
pxy[k*px.size()+l]= py[k]*px[l];
for( unsigned k=0; k<py.size(); k++)
for( unsigned l=0; l<px.size(); l++)
{
if( fabs(pxy[k*px.size() +l]) > 1e-14)
{
row_indices.push_back( i);
column_indices.push_back( (colsY[k])*g.nx()*g.Nx() +
colsX[l]);
values.push_back( negative ? - pxy[k*px.size()+l]
:  pxy[k*px.size()+l]);
}
}
}
}
cusp::coo_matrix<int, real_type, cusp::host_memory> A( x.size(),
g.size(), values.size());
A.row_indices = row_indices;
A.column_indices = column_indices;
A.values = values;

return A;
}




template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
const thrust::host_vector<real_type>& x,
const thrust::host_vector<real_type>& y,
const thrust::host_vector<real_type>& z,
const aRealTopology3d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
std::string method = "dg")
{
assert( x.size() == y.size());
assert( y.size() == z.size());
cusp::array1d<real_type, cusp::host_memory> values;
cusp::array1d<int, cusp::host_memory> row_indices;
cusp::array1d<int, cusp::host_memory> column_indices;

if( method == "dg")
{
std::vector<real_type> gauss_nodesx = g.dltx().abscissas();
std::vector<real_type> gauss_nodesy = g.dlty().abscissas();
std::vector<real_type> gauss_nodesz = g.dltz().abscissas();
dg::Operator<real_type> forwardx( g.dltx().forward());
dg::Operator<real_type> forwardy( g.dlty().forward());
dg::Operator<real_type> forwardz( g.dltz().forward());
for( int i=0; i<(int)x.size(); i++)
{
real_type X = x[i], Y = y[i], Z = z[i];
bool negative = false;
g.shift( negative,X,Y,Z, bcx, bcy, bcz);

real_type xnn = (X-g.x0())/g.hx();
real_type ynn = (Y-g.y0())/g.hy();
real_type znn = (Z-g.z0())/g.hz();
unsigned nn = (unsigned)floor(xnn);
unsigned mm = (unsigned)floor(ynn);
unsigned ll = (unsigned)floor(znn);
real_type xn = 2.*xnn - (real_type)(2*nn+1);
real_type yn = 2.*ynn - (real_type)(2*mm+1);
real_type zn = 2.*znn - (real_type)(2*ll+1);
if (nn==g.Nx()) {
nn-=1;
xn =1.;
}
if (mm==g.Ny()) {
mm-=1;
yn =1.;
}
if (ll==g.Nz()) {
ll-=1;
zn =1.;
}
int idxX =-1, idxY = -1, idxZ = -1;
for( unsigned k=0; k<g.nx(); k++)
{
if( fabs( xn - gauss_nodesx[k]) < 1e-14)
idxX = nn*g.nx() + k; 
}
for( unsigned k=0; k<g.ny(); k++)
{
if( fabs( yn - gauss_nodesy[k]) < 1e-14)
idxY = mm*g.ny() + k;  
}
for( unsigned k=0; k<g.nz(); k++)
{
if( fabs( zn - gauss_nodesz[k]) < 1e-14)
idxZ = ll*g.nz() + k;  
}
if( idxX >= 0 && idxY >= 0 && idxZ >= 0) 
{
row_indices.push_back(i);
column_indices.push_back((idxZ*g.Ny()*g.ny()+idxY)*g.Nx()*g.nx() + idxX);
if( !negative)
values.push_back( 1.);
else
values.push_back(-1.);
}
else if ( idxX < 0 && idxY >=0 && idxZ >= 0)
{
std::vector<real_type> px = detail::coefficients( xn, g.nx());
std::vector<real_type> pxF(g.nx(),0);
for( unsigned l=0; l<g.nx(); l++)
for( unsigned k=0; k<g.nx(); k++)
pxF[l]+= px[k]*forwardx(k,l);
for( unsigned l=0; l<g.nx(); l++)
{
row_indices.push_back( i);
column_indices.push_back( (idxZ*g.Ny()*g.ny() +
idxY)*g.Nx()*g.nx() + nn*g.nx() + l);
if( !negative)
values.push_back( pxF[l]);
else
values.push_back(-pxF[l]);
}
}
else if ( idxX >= 0 && idxY < 0 && idxZ >= 0)
{
std::vector<real_type> py = detail::coefficients( yn, g.ny());
std::vector<real_type> pyF(g.ny(),0);
for( unsigned l=0; l<g.ny(); l++)
for( unsigned k=0; k<g.ny(); k++)
pyF[l]+= py[k]*forwardy(k,l);
for( unsigned k=0; k<g.ny(); k++)
{
row_indices.push_back(i);
column_indices.push_back(((idxZ*g.Ny()+mm)*g.ny()+k)*g.Nx()*g.nx() + idxX);
if(!negative)
values.push_back( pyF[k]);
else
values.push_back(-pyF[k]);
}
}
else
{
std::vector<real_type> px = detail::coefficients( xn, g.nx()),
py = detail::coefficients( yn, g.ny()),
pz = detail::coefficients( zn, g.nz());
std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0), pzF( g.nz(), 0);
for( unsigned l=0; l<g.nx(); l++)
for( unsigned k=0; k<g.nx(); k++)
pxF[l]+= px[k]*forwardx(k,l);
for( unsigned l=0; l<g.ny(); l++)
for( unsigned k=0; k<g.ny(); k++)
pyF[l]+= py[k]*forwardy(k,l);
for( unsigned l=0; l<g.nz(); l++)
for( unsigned k=0; k<g.nz(); k++)
pzF[l]+= pz[k]*forwardz(k,l);
for( unsigned s=0; s<g.nz(); s++)
for( unsigned k=0; k<g.ny(); k++)
for( unsigned l=0; l<g.nx(); l++)
{
row_indices.push_back( i);
column_indices.push_back(
((((ll*g.nz()+s)*g.Ny()+mm)*g.ny()+k)*g.Nx()+nn)*g.nx()+l);
real_type pxyz = pzF[s]*pyF[k]*pxF[l];
if( !negative)
values.push_back( pxyz);
else
values.push_back(-pxyz);
}
}
}
}
else
{
unsigned points_per_line = 1;
if( method == "nearest")
points_per_line = 1;
else if( method == "linear")
points_per_line = 2;
else if( method == "cubic")
points_per_line = 4;
else
throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
RealGrid1d<real_type> gx(g.x0(), g.x1(), g.nx(), g.Nx(), bcx);
RealGrid1d<real_type> gy(g.y0(), g.y1(), g.ny(), g.Ny(), bcy);
RealGrid1d<real_type> gz(g.z0(), g.z1(), g.nz(), g.Nz(), bcz);
thrust::host_vector<real_type> absX = dg::create::abscissas( gx);
thrust::host_vector<real_type> absY = dg::create::abscissas( gy);
thrust::host_vector<real_type> absZ = dg::create::abscissas( gz);
for( unsigned i=0; i<x.size(); i++)
{
real_type X = x[i], Y = y[i], Z = z[i];
bool negative = false;
g.shift( negative, X, Y, Z, bcx, bcy, bcz);

thrust::host_vector<unsigned> colsX, colsY, colsZ;
std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
points_per_line, gx, absX, colsX);
std::vector<real_type> ys  = detail::choose_1d_abscissas( Y,
points_per_line, gy, absY, colsY);
std::vector<real_type> zs  = detail::choose_1d_abscissas( Z,
points_per_line, gz, absZ, colsZ);

std::vector<real_type> pxyz( points_per_line*points_per_line
*points_per_line);
std::vector<real_type> px = detail::lagrange( X, xs),
py = detail::lagrange( Y, ys),
pz = detail::lagrange( Z, zs);
for( unsigned m=0; m<pz.size(); m++)
for( unsigned k=0; k<py.size(); k++)
for( unsigned l=0; l<px.size(); l++)
pxyz[(m*py.size()+k)*px.size()+l]= pz[m]*py[k]*px[l];
for( unsigned m=0; m<pz.size(); m++)
for( unsigned k=0; k<py.size(); k++)
for( unsigned l=0; l<px.size(); l++)
{
if( fabs(pxyz[(m*py.size()+k)*px.size() +l]) > 1e-14)
{
row_indices.push_back( i);
column_indices.push_back( ((colsZ[m])*g.ny()*g.Ny() +
colsY[k])*g.nx()*g.Nx() + colsX[l]);
values.push_back( negative ?
-pxyz[(m*py.size()+k)*px.size()+l]
:  pxyz[(m*py.size()+k)*px.size()+l] );
}
}
}
}
cusp::coo_matrix<int, real_type, cusp::host_memory> A( x.size(), g.size(),
values.size());
A.row_indices = row_indices;
A.column_indices = column_indices;
A.values = values;

return A;
}

template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old, std::string method = "dg")
{
assert( g_new.x0() >= g_old.x0());
assert( g_new.x1() <= g_old.x1());
thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX1d, g_new);
return interpolation( pointsX, g_old, g_old.bcx(), method);

}
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old, std::string method = "dg")
{
assert( g_new.x0() >= g_old.x0());
assert( g_new.x1() <= g_old.x1());
assert( g_new.y0() >= g_old.y0());
assert( g_new.y1() <= g_old.y1());
thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX2d, g_new);

thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY2d, g_new);
return interpolation( pointsX, pointsY, g_old, g_old.bcx(), g_old.bcy(), method);

}

template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old, std::string method = "dg")
{
assert( g_new.x0() >= g_old.x0());
assert( g_new.x1() <= g_old.x1());
assert( g_new.y0() >= g_old.y0());
assert( g_new.y1() <= g_old.y1());
assert( g_new.z0() >= g_old.z0());
assert( g_new.z1() <= g_old.z1());
thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX3d, g_new);
thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY3d, g_new);
thrust::host_vector<real_type> pointsZ = dg::evaluate( dg::cooZ3d, g_new);
return interpolation( pointsX, pointsY, pointsZ, g_old, g_old.bcx(), g_old.bcy(), g_old.bcz(), method);

}
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology3d<real_type>& g_new, const aRealTopology2d<real_type>& g_old, std::string method = "dg")
{
assert( g_new.x0() >= g_old.x0());
assert( g_new.x1() <= g_old.x1());
assert( g_new.y0() >= g_old.y0());
assert( g_new.y1() <= g_old.y1());
thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX3d, g_new);
thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY3d, g_new);
return interpolation( pointsX, pointsY, g_old, g_old.bcx(), g_old.bcy(), method);

}


}



template<class real_type>
real_type interpolate(
dg::space sp,
const thrust::host_vector<real_type>& v,
real_type x,
const RealGrid1d<real_type>& g,
dg::bc bcx = dg::NEU)
{
assert( v.size() == g.size());
bool negative = false;
g.shift( negative, x, bcx);


real_type xnn = (x-g.x0())/g.h();
unsigned n = (unsigned)floor(xnn);

real_type xn =  2.*xnn - (real_type)(2*n+1);
if (n==g.N()) {
n-=1;
xn = 1.;
}
std::vector<real_type> px = create::detail::coefficients( xn, g.n());
if( sp == dg::xspace)
{
dg::Operator<real_type> forward( g.dlt().forward());
std::vector<real_type> pxF(g.n(),0);
for( unsigned l=0; l<g.n(); l++)
for( unsigned k=0; k<g.n(); k++)
pxF[l]+= px[k]*forward(k,l);
for( unsigned k=0; k<g.n(); k++)
px[k] = pxF[k];
}
unsigned cols = (n)*g.n();
real_type value = 0;
for( unsigned j=0; j<g.n(); j++)
{
if(negative)
value -= v[cols + j]*px[j];
else
value += v[cols + j]*px[j];
}
return value;
}


template<class real_type>
real_type interpolate(
dg::space sp,
const thrust::host_vector<real_type>& v,
real_type x, real_type y,
const aRealTopology2d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU )
{
assert( v.size() == g.size());
bool negative = false;
g.shift( negative, x,y, bcx, bcy);


real_type xnn = (x-g.x0())/g.hx();
real_type ynn = (y-g.y0())/g.hy();
unsigned n = (unsigned)floor(xnn);
unsigned m = (unsigned)floor(ynn);

real_type xn =  2.*xnn - (real_type)(2*n+1);
real_type yn =  2.*ynn - (real_type)(2*m+1);
if (n==g.Nx()) {
n-=1;
xn = 1.;
}
if (m==g.Ny()) {
m-=1;
yn =1.;
}
std::vector<real_type> px = create::detail::coefficients( xn, g.nx()),
py = create::detail::coefficients( yn, g.ny());
if( sp == dg::xspace)
{
dg::Operator<real_type> forwardx( g.dltx().forward());
dg::Operator<real_type> forwardy( g.dlty().forward());
std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0);
for( unsigned l=0; l<g.nx(); l++)
for( unsigned k=0; k<g.nx(); k++)
pxF[l]+= px[k]*forwardx(k,l);
for( unsigned l=0; l<g.ny(); l++)
for( unsigned k=0; k<g.ny(); k++)
pyF[l]+= py[k]*forwardy(k,l);
px = pxF, py = pyF;
}
unsigned cols = (m)*g.Nx()*g.ny()*g.nx() + (n)*g.nx();
real_type value = 0;
for( unsigned i=0; i<g.ny(); i++)
for( unsigned j=0; j<g.nx(); j++)
{
if(negative)
value -= v[cols + i*g.Nx()*g.nx() + j]*px[j]*py[i];
else
value += v[cols + i*g.Nx()*g.nx() + j]*px[j]*py[i];
}
return value;
}

} 
