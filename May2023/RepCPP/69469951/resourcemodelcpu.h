


#pragma once


#include "resourcemodel.h"

template< typename NodeOrderT,
typename CPUOrderT >
class ResourceModelCPU
{
public:
ResourceModelCPU( CPUOrderT order = 0 ): traceGlobalOrder( order )
{}

~ResourceModelCPU()
{}

bool operator==( const ResourceModelCPU< NodeOrderT, CPUOrderT >& other ) const
{
return traceGlobalOrder == other.traceGlobalOrder;
}

protected:
CPUOrderT traceGlobalOrder;

private:
template< typename NodeOrderU, typename CPUOrderU > friend class ResourceModel;

};
