#pragma once

#include "grid.h"
#include "dx.h"



namespace dg{



namespace create{




template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology2d<real_type>& g, bc bcx, direction dir = centered)
{
EllSparseBlockMat<real_type> dx;
dx = dx_normed( g.nx(), g.Nx(), g.hx(), bcx, dir);
dx.set_left_size( g.ny()*g.Ny());
return dx;
}


template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology2d<real_type>& g, direction dir = centered) {
return dx( g, g.bcx(), dir);
}


template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology2d<real_type>& g, bc bcy, direction dir = centered)
{
EllSparseBlockMat<real_type> dy;
dy = dx_normed( g.ny(), g.Ny(), g.hy(), bcy, dir);
dy.set_right_size( g.nx()*g.Nx());
return dy;
}


template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology2d<real_type>& g, direction dir = centered){
return dy( g, g.bcy(), dir);
}


template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology2d<real_type>& g, bc bcx)
{
EllSparseBlockMat<real_type> jx;
jx = jump( g.nx(), g.Nx(), g.hx(), bcx);
jx.set_left_size( g.ny()*g.Ny());
return jx;
}


template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology2d<real_type>& g, bc bcy)
{
EllSparseBlockMat<real_type> jy;
jy = jump( g.ny(), g.Ny(), g.hy(), bcy);
jy.set_right_size( g.nx()*g.Nx());
return jy;
}


template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology2d<real_type>& g) {
return jumpX( g, g.bcx());
}


template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology2d<real_type>& g) {
return jumpY( g, g.bcy());
}


template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology3d<real_type>& g, bc bcx)
{
EllSparseBlockMat<real_type> jx;
jx = jump( g.nx(), g.Nx(), g.hx(), bcx);
jx.set_left_size( g.ny()*g.Ny()*g.nz()*g.Nz());
return jx;
}


template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology3d<real_type>& g, bc bcy)
{
EllSparseBlockMat<real_type> jy;
jy = jump( g.ny(), g.Ny(), g.hy(), bcy);
jy.set_right_size( g.nx()*g.Nx());
jy.set_left_size( g.nz()*g.Nz());
return jy;
}


template<class real_type>
EllSparseBlockMat<real_type> jumpZ( const aRealTopology3d<real_type>& g, bc bcz)
{
EllSparseBlockMat<real_type> jz;
jz = jump( g.nz(), g.Nz(), g.hz(), bcz);
jz.set_right_size( g.nx()*g.Nx()*g.ny()*g.Ny());
return jz;
}


template<class real_type>
EllSparseBlockMat<real_type> jumpX( const aRealTopology3d<real_type>& g) {
return jumpX( g, g.bcx());
}


template<class real_type>
EllSparseBlockMat<real_type> jumpY( const aRealTopology3d<real_type>& g) {
return jumpY( g, g.bcy());
}


template<class real_type>
EllSparseBlockMat<real_type> jumpZ( const aRealTopology3d<real_type>& g) {
return jumpZ( g, g.bcz());
}



template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology3d<real_type>& g, bc bcx, direction dir = centered)
{
EllSparseBlockMat<real_type> dx;
dx = dx_normed( g.nx(), g.Nx(), g.hx(), bcx, dir);
dx.set_left_size( g.ny()*g.Ny()*g.nz()*g.Nz());
return dx;
}


template<class real_type>
EllSparseBlockMat<real_type> dx( const aRealTopology3d<real_type>& g, direction dir = centered) {
return dx( g, g.bcx(), dir);
}


template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology3d<real_type>& g, bc bcy, direction dir = centered)
{
EllSparseBlockMat<real_type> dy;
dy = dx_normed( g.ny(), g.Ny(), g.hy(), bcy, dir);
dy.set_right_size( g.nx()*g.Nx());
dy.set_left_size( g.nz()*g.Nz());
return dy;
}


template<class real_type>
EllSparseBlockMat<real_type> dy( const aRealTopology3d<real_type>& g, direction dir = centered){
return dy( g, g.bcy(), dir);
}


template<class real_type>
EllSparseBlockMat<real_type> dz( const aRealTopology3d<real_type>& g, bc bcz, direction dir = centered)
{
EllSparseBlockMat<real_type> dz;
dz = dx_normed( g.nz(), g.Nz(), g.hz(), bcz, dir);
dz.set_right_size( g.nx()*g.ny()*g.Nx()*g.Ny());
return dz;

}


template<class real_type>
EllSparseBlockMat<real_type> dz( const aRealTopology3d<real_type>& g, direction dir = centered){
return dz( g, g.bcz(), dir);
}




} 

} 

