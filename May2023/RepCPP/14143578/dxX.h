#pragma once

#include <cassert>

#include "gridX.h"
#include "dx.h"
#include "weightsX.h"
#include "dg/backend/sparseblockmat.h"


namespace dg
{
namespace create
{

template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGridX1d<real_type>& g, bc bcx, direction dir = centered)
{
if( g.outer_N() == 0) return dx( g.grid(), dg::PER, dir);
EllSparseBlockMat<real_type> DX = dx( g.grid(), bcx, dir);
for( int i=0; i<DX.blocks_per_line; i++)
{
if( DX.cols_idx[DX.blocks_per_line*(g.outer_N()-1)+i] == (int)g.outer_N())
DX.cols_idx[DX.blocks_per_line*(g.outer_N()-1)+i] += g.inner_N();
if( DX.cols_idx[DX.blocks_per_line*(g.outer_N())+i] == (int)g.outer_N()-1)
DX.cols_idx[DX.blocks_per_line*(g.outer_N())+i] += g.inner_N();
if( DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N()-1)+i] == (int)(g.N()-g.outer_N()))
DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N()-1)+i] -= g.inner_N();
if( DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N())+i] == (int)(g.N()-g.outer_N()-1))
DX.cols_idx[DX.blocks_per_line*(g.N()-g.outer_N())+i] -= g.inner_N();
}
return DX;
}


template<class real_type>
EllSparseBlockMat<real_type> dx( const RealGridX1d<real_type>& g, direction dir = centered)
{
return dx( g, g.bcx(), dir);
}

template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGridX1d<real_type>& g, bc bcx)
{
if( g.outer_N() == 0) return jump( g.n(), g.N(), g.h(), dg::PER);
EllSparseBlockMat<real_type> J = jump( g.n(),g.N(),g.h(), bcx);
for( int i=0; i<J.blocks_per_line; i++)
{
if( J.cols_idx[J.blocks_per_line*(g.outer_N()-1)+i] == (int)g.outer_N())
J.cols_idx[J.blocks_per_line*(g.outer_N()-1)+i] += g.inner_N();
if( J.cols_idx[J.blocks_per_line*(g.outer_N())+i] == (int)g.outer_N()-1)
J.cols_idx[J.blocks_per_line*(g.outer_N())+i] += g.inner_N();
if( J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N()-1)+i] == (int)(g.N()-g.outer_N()))
J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N()-1)+i] -= g.inner_N();
if( J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N())+i] == (int)(g.N()-g.outer_N()-1))
J.cols_idx[J.blocks_per_line*(g.N()-g.outer_N())+i] -= g.inner_N();
}
return J;
}

template<class real_type>
EllSparseBlockMat<real_type> jump( const RealGridX1d<real_type>& g)
{
return jump( g, g.bcx());
}

} 
} 

