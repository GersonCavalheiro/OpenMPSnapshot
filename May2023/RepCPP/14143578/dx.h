#pragma once

#include <cassert>

#include "dg/backend/sparseblockmat.h"
#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "weights.h"


namespace dg
{
namespace create
{



template<class real_type>
EllSparseBlockMat<real_type> dx_symm(int n, int N, real_type h, bc bcx)
{

Operator<real_type> l = create::lilj<real_type>(n);
Operator<real_type> r = create::rirj<real_type>(n);
Operator<real_type> lr = create::lirj<real_type>(n);
Operator<real_type> rl = create::rilj<real_type>(n);
Operator<real_type> d = create::pidxpj<real_type>(n);
Operator<real_type> t = create::pipj_inv<real_type>(n);
t *= 2./h;

Operator< real_type> a = 1./2.*t*(d-d.transpose());
Operator<real_type> a_bound_right(a), a_bound_left(a);
if( bcx == DIR || bcx == DIR_NEU )
a_bound_left += 0.5*t*l;
else if (bcx == NEU || bcx == NEU_DIR)
a_bound_left -= 0.5*t*l;
if( bcx == DIR || bcx == NEU_DIR)
a_bound_right -= 0.5*t*r;
else if( bcx == NEU || bcx == DIR_NEU)
a_bound_right += 0.5*t*r;
if( bcx == PER ) 
a_bound_left = a_bound_right = a;
Operator<real_type> b = t*(1./2.*rl);
Operator<real_type> bp = t*(-1./2.*lr); 
RealGrid1d<real_type> g( 0,1, n, N);
Operator<real_type> backward=g.dlt().backward();
Operator<real_type> forward=g.dlt().forward();
a = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
bp = backward*bp*forward;
if( bcx != PER)
{
EllSparseBlockMat<real_type> A(N, N, 3, 6, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
A.data[(2*n+i)*n+j] = b(i,j);
A.data[(3*n+i)*n+j] = a_bound_left(i,j);
A.data[(4*n+i)*n+j] = a_bound_right(i,j);
A.data[(5*n+i)*n+j] = 0; 
}
A.data_idx[0*3+0] = 3; 
A.cols_idx[0*3+0] = 0;
A.data_idx[0*3+1] = 2; 
A.cols_idx[0*3+1] = 1;
A.data_idx[0*3+2] = 5; 
A.cols_idx[0*3+2] = 1; 
for( int i=1; i<N-1; i++)
for( int d=0; d<3; d++)
{
A.data_idx[i*3+d] = d; 
A.cols_idx[i*3+d] = i+d-1;
}
A.data_idx[(N-1)*3+0] = 0; 
A.cols_idx[(N-1)*3+0] = N-2;
A.data_idx[(N-1)*3+1] = 4; 
A.cols_idx[(N-1)*3+1] = N-1;
A.data_idx[(N-1)*3+2] = 5; 
A.cols_idx[(N-1)*3+2] = N-1; 
return A;

}
else 
{
EllSparseBlockMat<real_type> A(N, N, 3, 3, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
A.data[(2*n+i)*n+j] = b(i,j);
}
for( int i=0; i<N; i++)
for( int d=0; d<3; d++)
{
A.data_idx[i*3+d] = d; 
A.cols_idx[i*3+d] = (i+d-1+N)%N;
}
return A;
}
};


template<class real_type>
EllSparseBlockMat<real_type> dx_plus( int n, int N, real_type h, bc bcx )
{

Operator<real_type> l = create::lilj<real_type>(n);
Operator<real_type> r = create::rirj<real_type>(n);
Operator<real_type> lr = create::lirj<real_type>(n);
Operator<real_type> rl = create::rilj<real_type>(n);
Operator<real_type> d = create::pidxpj<real_type>(n);
Operator<real_type> t = create::pipj_inv<real_type>(n);
t *= 2./h;
Operator<real_type>  a = t*(-l-d.transpose());
Operator<real_type> a_bound_left = a; 
Operator<real_type> a_bound_right = a; 
if( bcx == dg::DIR || bcx == dg::DIR_NEU)
a_bound_left = t*(-d.transpose());
if( bcx == dg::NEU || bcx == dg::DIR_NEU)
a_bound_right = t*(d);
Operator<real_type> b = t*rl;
RealGrid1d<real_type> g( 0,1, n, N);
Operator<real_type> backward=g.dlt().backward();
Operator<real_type> forward=g.dlt().forward();
a = backward*a*forward, a_bound_left = backward*a_bound_left*forward;
b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
if( bcx != PER)
{
EllSparseBlockMat<real_type> A(N, N, 2, 5, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = a(i,j);
A.data[(1*n+i)*n+j] = b(i,j);
A.data[(2*n+i)*n+j] = a_bound_left(i,j);
A.data[(3*n+i)*n+j] = a_bound_right(i,j);
A.data[(4*n+i)*n+j] = 0; 
}
A.data_idx[0*2+0] = 2; 
A.cols_idx[0*2+0] = 0;
A.data_idx[0*2+1] = 1; 
A.cols_idx[0*2+1] = 1;
for( int i=1; i<N-1; i++) 
for( int d=0; d<2; d++)
{
A.data_idx[i*2+d] = d; 
A.cols_idx[i*2+d] = i+d; 
}
A.data_idx[(N-1)*2+0] = 3; 
A.cols_idx[(N-1)*2+0] = N-1;
A.data_idx[(N-1)*2+1] = 4; 
A.cols_idx[(N-1)*2+1] = N-1; 
return A;

}
else 
{
EllSparseBlockMat<real_type> A(N, N, 2, 2, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = a(i,j);
A.data[(1*n+i)*n+j] = b(i,j);
}
for( int i=0; i<N; i++)
for( int d=0; d<2; d++)
{
A.data_idx[i*2+d] = d; 
A.cols_idx[i*2+d] = (i+d+N)%N;
}
return A;
}
};


template<class real_type>
EllSparseBlockMat<real_type> dx_minus( int n, int N, real_type h, bc bcx )
{
Operator<real_type> l = create::lilj<real_type>(n);
Operator<real_type> r = create::rirj<real_type>(n);
Operator<real_type> lr = create::lirj<real_type>(n);
Operator<real_type> rl = create::rilj<real_type>(n);
Operator<real_type> d = create::pidxpj<real_type>(n);
Operator<real_type> t = create::pipj_inv<real_type>(n);
t *= 2./h;
Operator<real_type>  a = t*(l+d);
Operator<real_type> a_bound_right = a; 
Operator<real_type> a_bound_left = a; 
if( bcx == dg::DIR || bcx == dg::NEU_DIR)
a_bound_right = t*(-d.transpose());
if( bcx == dg::NEU || bcx == dg::NEU_DIR)
a_bound_left = t*d;
Operator<real_type> bp = -t*lr;
RealGrid1d<real_type> g( 0,1, n, N);
Operator<real_type> backward=g.dlt().backward();
Operator<real_type> forward=g.dlt().forward();
a  = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
bp = backward*bp*forward, a_bound_right = backward*a_bound_right*forward;

if(bcx != dg::PER)
{
EllSparseBlockMat<real_type> A(N, N, 2, 5, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
A.data[(2*n+i)*n+j] = a_bound_left(i,j);
A.data[(3*n+i)*n+j] = a_bound_right(i,j);
A.data[(4*n+i)*n+j] = 0; 
}
A.data_idx[0*2+0] = 2; 
A.cols_idx[0*2+0] = 0;
A.data_idx[0*2+1] = 4; 
A.cols_idx[0*2+1] = 0; 
for( int i=1; i<N-1; i++) 
for( int d=0; d<2; d++)
{
A.data_idx[i*2+d] = d; 
A.cols_idx[i*2+d] = i+d-1;
}
A.data_idx[(N-1)*2+0] = 0; 
A.cols_idx[(N-1)*2+0] = N-2;
A.data_idx[(N-1)*2+1] = 3; 
A.cols_idx[(N-1)*2+1] = N-1;
return A;

}
else 
{
EllSparseBlockMat<real_type> A(N, N, 2, 2, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
}
for( int i=0; i<N; i++)
for( int d=0; d<2; d++)
{
A.data_idx[i*2+d] = d; 
A.cols_idx[i*2+d] = (i+d-1+N)%N;  
}
return A;
}
};


template<class real_type>
EllSparseBlockMat<real_type> jump( int n, int N, real_type h, bc bcx)
{
Operator<real_type> l = create::lilj<real_type>(n);
Operator<real_type> r = create::rirj<real_type>(n);
Operator<real_type> lr = create::lirj<real_type>(n);
Operator<real_type> rl = create::rilj<real_type>(n);
Operator<real_type> a = l+r;
Operator<real_type> a_bound_left = a;
if( bcx == NEU || bcx == NEU_DIR)
a_bound_left = r;
Operator<real_type> a_bound_right = a; 
if( bcx == NEU || bcx == DIR_NEU)
a_bound_right = l;
Operator<real_type> b = -rl;
Operator<real_type> bp = -lr;
Operator<real_type> t = create::pipj_inv<real_type>(n);
t *= 2./h;
RealGrid1d<real_type> g( 0,1, n, N);
Operator<real_type> backward=g.dlt().backward();
Operator<real_type> forward=g.dlt().forward();
a = backward*t*a*forward, a_bound_left  = backward*t*a_bound_left*forward;
b = backward*t*b*forward, a_bound_right = backward*t*a_bound_right*forward;
bp = backward*t*bp*forward;
if(bcx != dg::PER)
{
EllSparseBlockMat<real_type> A(N, N, 3, 6, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
A.data[(2*n+i)*n+j] = b(i,j);
A.data[(3*n+i)*n+j] = a_bound_left(i,j);
A.data[(4*n+i)*n+j] = a_bound_right(i,j);
A.data[(5*n+i)*n+j] = 0; 
}
A.data_idx[0*3+0] = 3; 
A.cols_idx[0*3+0] = 0;
A.data_idx[0*3+1] = 2; 
A.cols_idx[0*3+1] = 1;
A.data_idx[0*3+2] = 5; 
A.cols_idx[0*3+2] = 1; 
for( int i=1; i<N-1; i++) 
for( int d=0; d<3; d++)
{
A.data_idx[i*3+d] = d; 
A.cols_idx[i*3+d] = i+d-1;
}
A.data_idx[(N-1)*3+0] = 0; 
A.cols_idx[(N-1)*3+0] = N-2;
A.data_idx[(N-1)*3+1] = 4; 
A.cols_idx[(N-1)*3+1] = N-1;
A.data_idx[(N-1)*3+2] = 5; 
A.cols_idx[(N-1)*3+2] = N-1; 
return A;

}
else 
{
EllSparseBlockMat<real_type> A(N, N, 3, 3, n);
for( int i=0; i<n; i++)
for( int j=0; j<n; j++)
{
A.data[(0*n+i)*n+j] = bp(i,j);
A.data[(1*n+i)*n+j] = a(i,j);
A.data[(2*n+i)*n+j] = b(i,j);
}
for( int i=0; i<N; i++)
for( int d=0; d<3; d++)
{
A.data_idx[i*3+d] = d; 
A.cols_idx[i*3+d] = (i+d-1+N)%N;
}
return A;
}
};


template<class real_type>
EllSparseBlockMat<real_type> dx_normed( int n, int N, real_type h, bc bcx, direction dir )
{
if( dir == centered)
return create::dx_symm(n, N, h, bcx);
else if (dir == forward)
return create::dx_plus(n, N, h, bcx);
else if (dir == backward)
return create::dx_minus(n, N, h, bcx);
return EllSparseBlockMat<real_type>();
}


template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGrid1d<real_type>& g, bc bcx, direction dir = centered)
{
return dx_normed( g.n(), g.N(), g.h(), bcx, dir);
}


template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGrid1d<real_type>& g, direction dir = centered)
{
return dx( g, g.bcx(), dir);
}


template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGrid1d<real_type>& g, bc bcx)
{
return jump( g.n(), g.N(), g.h(), bcx);
}

template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGrid1d<real_type>& g)
{
return jump( g, g.bcx());
}


} 
} 

