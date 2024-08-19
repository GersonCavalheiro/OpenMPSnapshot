#pragma once

#include "weights.h"
#include "gridX.h"



namespace dg{
namespace create{


template<class real_type>
thrust::host_vector<real_type> weights( const dg::RealGridX1d<real_type>& g) { return weights( g.grid()); }
template<class real_type>
thrust::host_vector<real_type> inv_weights( const RealGridX1d<real_type>& g) { return inv_weights( g.grid()); }

template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX2d<real_type>& g) { return weights( g.grid()); }
template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX2d<real_type>& g) { return inv_weights( g.grid()); }

template<class real_type>
thrust::host_vector<real_type> weights( const aRealTopologyX3d<real_type>& g) { return weights(g.grid()); }

template<class real_type>
thrust::host_vector<real_type> inv_weights( const aRealTopologyX3d<real_type>& g) { return inv_weights(g.grid()); }

}
}
