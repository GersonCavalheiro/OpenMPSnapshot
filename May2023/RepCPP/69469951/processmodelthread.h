


#pragma once



#include "processmodel.h"

template< typename ApplOrderT,
typename TaskOrderT,
typename ThreadOrderT,
typename NodeOrderT >
class ProcessModelThread
{

public:
ProcessModelThread( ThreadOrderT order = 0,
NodeOrderT node = 0 ):
traceGlobalOrder( order ),
nodeExecution( node )
{}

~ProcessModelThread()
{}

bool operator==( const ProcessModelThread< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT >& other ) const
{
return traceGlobalOrder == other.traceGlobalOrder &&
nodeExecution    == other.nodeExecution;
}

NodeOrderT getNodeExecution() const { return nodeExecution; }

protected:
ThreadOrderT traceGlobalOrder;
NodeOrderT   nodeExecution;

private:
template< typename ApplOrderU, typename TaskOrderU, typename ThreadOrderU, typename NodeOrderU > friend class ProcessModel;

};


