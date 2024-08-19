#pragma once

#include "gridX.h"
#include "dxX.h"
#include "../blas.h"


namespace dg{

template<class Matrix>
struct Composite
{
Composite( ):m1(), m2(), dual(false){ }
template<class Matrix2>
Composite( const Composite<Matrix2>& src):m1(src.m1), m2(src.m2), dual(src.dual){}
Composite( const Matrix& m):m1(m), m2(m), dual(false){ }
Composite( const Matrix& m1, const Matrix& m2):m1(m1), m2(m2), dual(true){ }
template<class Matrix2>
Composite& operator=( const Composite<Matrix2>& src){ Composite c(src);
*this = c; return *this;}
template< class ContainerType1, class ContainerType2>
void symv( const ContainerType1& v1, ContainerType2& v2) const
{
dg::blas2::symv( m1, v1, v2); 
if( dual)
dg::blas2::symv( m2, v1, v2); 
}
template< class ContainerType>
void symv( get_value_type<ContainerType> alpha, const  ContainerType& v1, get_value_type<ContainerType> beta, ContainerType& v2) const
{
dg::blas2::symv( alpha, m1, v1, beta, v2); 
if( dual)
dg::blas2::symv( alpha, m2, v1, beta, v2); 
}
void display( std::ostream& os = std::cout) const
{
if( dual)
{
os << " dual matrix: \n";
os << " INNER MATRIX\n";
m1.display( os);
os << " OUTER MATRIX\n";
m2.display( os);
}
else
{
os << "single matrix: \n";
m1.display(os);
}
}
Matrix m1, m2;
bool dual;
};
template <class Matrix>
struct TensorTraits<Composite<Matrix> >
{
using value_type = get_value_type<Matrix>;
using tensor_category = SelfMadeMatrixTag;
};



namespace create{




template<class real_type>
Composite<EllSparseBlockMat<real_type> > dx( const aRealTopologyX2d<real_type>& g, bc bcx, direction dir = centered)
{
EllSparseBlockMat<real_type>  dx;
dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
dx.set_left_size( g.n()*g.Ny());
return dx;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dx( const aRealTopologyX2d<real_type>& g, direction dir = centered) { return dx( g, g.bcx(), dir);}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dy( const aRealTopologyX2d<real_type>& g, bc bcy, direction dir = centered)
{
EllSparseBlockMat<real_type>  dy_inner, dy_outer;
RealGridX1d<real_type> g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
RealGrid1d<real_type> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
dy_inner = dx( g1d_inner, bcy, dir);
dy_outer = dx( g1d_outer, bcy, dir);
dy_inner.right_size = g.n()*g.Nx();
dy_inner.right_range[0] = 0;
dy_inner.right_range[1] = g.n()*g.inner_Nx();
dy_outer.right_range[0] = g.n()*g.inner_Nx();
dy_outer.right_range[1] = g.n()*g.Nx();
dy_outer.right_size = g.n()*g.Nx();

Composite<EllSparseBlockMat<real_type> > c( dy_inner, dy_outer);
return c;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dy( const aRealTopologyX2d<real_type>& g, direction dir = centered){ return dy( g, g.bcy(), dir);}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpX( const aRealTopologyX2d<real_type>& g, bc bcx)
{
EllSparseBlockMat<real_type>  jx;
jx = jump( g.n(), g.Nx(), g.hx(), bcx);
jx.set_left_size( g.n()*g.Ny());
return jx;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpY( const aRealTopologyX2d<real_type>& g, bc bcy)
{
EllSparseBlockMat<real_type>  jy_inner, jy_outer;
RealGridX1d<real_type> g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
RealGrid1d<real_type> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
jy_inner = jump( g1d_inner, bcy);
jy_outer = jump( g1d_outer, bcy);
jy_inner.right_size = g.n()*g.Nx();
jy_inner.right_range[0] = 0;
jy_inner.right_range[1] = g.n()*g.inner_Nx();
jy_outer.right_range[0] = g.n()*g.inner_Nx();
jy_outer.right_range[1] = g.n()*g.Nx();
jy_outer.right_size = g.n()*g.Nx();

Composite<EllSparseBlockMat<real_type> > c( jy_inner, jy_outer);
return c;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpX( const aRealTopologyX2d<real_type>& g)
{
return jumpX( g, g.bcx());
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpY( const aRealTopologyX2d<real_type>& g)
{
return jumpY( g, g.bcy());
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpX( const aRealTopologyX3d<real_type>& g, bc bcx)
{
EllSparseBlockMat<real_type>  jx;
jx = jump( g.n(), g.Nx(), g.hx(), bcx);
jx.set_left_size( g.n()*g.Ny()*g.Nz());
return jx;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpY( const aRealTopologyX3d<real_type>& g, bc bcy)
{
EllSparseBlockMat<real_type>  jy_inner, jy_outer;
RealGridX1d<real_type> g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
RealGrid1d<real_type> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
jy_inner = jump( g1d_inner, bcy);
jy_outer = jump( g1d_outer, bcy);
jy_inner.right_size = g.n()*g.Nx();
jy_inner.right_range[0] = 0;
jy_inner.right_range[1] = g.n()*g.inner_Nx();
jy_outer.right_range[0] = g.n()*g.inner_Nx();
jy_outer.right_range[1] = g.n()*g.Nx();
jy_outer.right_size = g.n()*g.Nx();
jy_inner.left_size = g.Nz();
jy_outer.left_size = g.Nz();

Composite<EllSparseBlockMat<real_type> > c( jy_inner, jy_outer);
return c;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpZ( const aRealTopologyX3d<real_type>& g, bc bcz)
{
EllSparseBlockMat<real_type>  jz;
jz = jump( 1, g.Nz(), g.hz(), bcz);
jz.set_right_size( g.n()*g.Nx()*g.n()*g.Ny());
return jz;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpX( const aRealTopologyX3d<real_type>& g)
{
return jumpX( g, g.bcx());
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpY( const aRealTopologyX3d<real_type>& g)
{
return jumpY( g, g.bcy());
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > jumpZ( const aRealTopologyX3d<real_type>& g)
{
return jumpZ( g, g.bcz());
}



template<class real_type>
Composite<EllSparseBlockMat<real_type> > dx( const aRealTopologyX3d<real_type>& g, bc bcx, direction dir = centered)
{
EllSparseBlockMat<real_type>  dx;
dx = dx_normed( g.n(), g.Nx(), g.hx(), bcx, dir);
dx.set_left_size( g.n()*g.Ny()*g.Nz());
return dx;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dx( const aRealTopologyX3d<real_type>& g, direction dir = centered) { return dx( g, g.bcx(), dir);}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dy( const aRealTopologyX3d<real_type>& g, bc bcy, direction dir = centered)
{
EllSparseBlockMat<real_type>  dy_inner, dy_outer;
RealGridX1d<real_type> g1d_inner( g.y0(), g.y1(), g.fy(), g.n(), g.Ny(), bcy);
RealGrid1d<real_type> g1d_outer( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
dy_inner = dx( g1d_inner, bcy, dir);
dy_outer = dx( g1d_outer, bcy, dir);
dy_inner.right_size = g.n()*g.Nx();
dy_inner.right_range[0] = 0;
dy_inner.right_range[1] = g.n()*g.inner_Nx();
dy_outer.right_range[0] = g.n()*g.inner_Nx();
dy_outer.right_range[1] = g.n()*g.Nx();
dy_outer.right_size = g.n()*g.Nx();
dy_inner.left_size = g.Nz();
dy_outer.left_size = g.Nz();

Composite<EllSparseBlockMat<real_type> > c( dy_inner, dy_outer);
return c;
}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dy( const aRealTopologyX3d<real_type>& g, direction dir = centered){ return dy( g, g.bcy(), dir);}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dz( const aRealTopologyX3d<real_type>& g, bc bcz, direction dir = centered)
{
EllSparseBlockMat<real_type>  dz;
dz = dx_normed( 1, g.Nz(), g.hz(), bcz, dir);
dz.set_right_size( g.n()*g.n()*g.Nx()*g.Ny());
return dz;

}


template<class real_type>
Composite<EllSparseBlockMat<real_type> > dz( const aRealTopologyX3d<real_type>& g, direction dir = centered){ return dz( g, g.bcz(), dir);}




} 

} 

