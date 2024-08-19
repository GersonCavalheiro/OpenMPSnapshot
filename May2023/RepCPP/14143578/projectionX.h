#pragma once
#include "projection.h"
#include "gridX.h"


namespace dg{
namespace create{

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const RealGridX1d<real_type>& g_new, const RealGridX1d<real_type>& g_old,std::string method = "dg") {
return projection(g_new.grid(), g_old.grid(),method);
}

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX2d<real_type>& g_new, const aRealTopologyX2d<real_type>& g_old,std::string method = "dg") {
return projection(g_new.grid(), g_old.grid(),method);
}

template<class real_type>
cusp::coo_matrix< int, real_type, cusp::host_memory> projection( const aRealTopologyX3d<real_type>& g_new, const aRealTopologyX3d<real_type>& g_old,std::string method = "dg") {
return projection(g_new.grid(), g_old.grid(),method);
}


}
}
