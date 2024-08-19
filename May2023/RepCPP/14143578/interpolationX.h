#pragma once

#include "interpolation.h"
#include "gridX.h"



namespace dg{

namespace create{

template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const RealGridX1d<real_type>& g)
{
return interpolation( x, g.grid());
}


template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const aRealTopologyX2d<real_type>& g , dg::bc globalbcz = dg::NEU)
{
return interpolation( x,y, g.grid(), globalbcz);
}




template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const thrust::host_vector<real_type>& x, const thrust::host_vector<real_type>& y, const thrust::host_vector<real_type>& z, const aRealTopologyX3d<real_type>& g, dg::bc globalbcz= dg::NEU)
{
return interpolation( x,y,z, g.grid(), globalbcz);
}


template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old)
{
return interpolation( g_new.grid(), g_old.grid());
}

template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old)
{
return interpolation( g_new.grid(), g_old.grid());
}


template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old)
{
return interpolation( g_new.grid(), g_old.grid());
}


template<class real_type>
thrust::host_vector<real_type> forward_transform( const thrust::host_vector<real_type>& in, const aRealTopologyX2d<real_type>& g)
{
return forward_transform( in, g.grid());
}
}


template<class real_type>
real_type interpolate(
dg::space sp,
const thrust::host_vector<real_type>& v,
real_type x, real_type y,
const aRealTopologyX2d<real_type>& g,
dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU )
{
return interpolate( sp, v, x, y, g.grid(), bcx, bcy);
}

} 
