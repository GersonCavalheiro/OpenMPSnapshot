


#pragma once


#include <vector>
#include "resourcemodelcpu.h"
#include "resourcemodel.h"

template< typename NodeOrderT,
typename CPUOrderT >
class ResourceModelNode
{
public:
ResourceModelNode( NodeOrderT order = 0 ) : traceGlobalOrder( order )
{}

~ResourceModelNode()
{}

bool operator==( const ResourceModelNode< NodeOrderT, CPUOrderT >& other ) const
{
return traceGlobalOrder == other.traceGlobalOrder &&
CPUs             == other.CPUs;
}

size_t size() const { return CPUs.size(); }

protected:
NodeOrderT traceGlobalOrder;
std::vector< ResourceModelCPU< NodeOrderT, CPUOrderT > > CPUs;

private:
template< typename NodeOrderU, typename CPUOrderU > friend class ResourceModel;

};

