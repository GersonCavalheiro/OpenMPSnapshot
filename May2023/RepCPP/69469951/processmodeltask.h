


#pragma once


#include <vector>
#include "processmodelthread.h"
#include "processmodel.h"

template< typename ApplOrderT,
typename TaskOrderT,
typename ThreadOrderT,
typename NodeOrderT >
class ProcessModelTask
{

public:
ProcessModelTask( TaskOrderT order = 0 ): traceGlobalOrder( order )
{}

~ProcessModelTask()
{}

bool operator==( const ProcessModelTask< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT >& other ) const
{
return traceGlobalOrder == other.traceGlobalOrder &&
threads          == other.threads;
}

size_t size() const { return threads.size(); }

NodeOrderT getNodeExecution() const { return threads[ 0 ].getNodeExecution(); }

protected:
TTaskOrder traceGlobalOrder;
std::vector< ProcessModelThread< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > > threads;

private:
template< typename ApplOrderU, typename TaskOrderU, typename ThreadOrderU, typename NodeOrderU > friend class ProcessModel;

};


