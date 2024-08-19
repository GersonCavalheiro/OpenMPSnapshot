#pragma once

#include "gridX.h"
#include "evaluation.h"



namespace dg
{

namespace create{
template<class real_type>
thrust::host_vector<real_type> abscissas( const RealGridX1d<real_type>& g)
{
return abscissas(g.grid());
}
}


template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const RealGridX1d<real_type>& g)
{
return evaluate( f, g.grid());
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const RealGridX1d<real_type>& g)
{
return evaluate( *f, g.grid());
};

template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aRealTopologyX2d<real_type>& g)
{
return evaluate( f, g.grid());
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aRealTopologyX2d<real_type>& g)
{
return evaluate( *f, g.grid());
};

template< class TernaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aRealTopologyX3d<real_type>& g)
{
return evaluate( f, g.grid());
};
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aRealTopologyX3d<real_type>& g)
{
return evaluate( *f, g.grid());
};

}

