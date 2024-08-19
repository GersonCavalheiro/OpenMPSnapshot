


#pragma once


#include <vector>
#include "processmodeltask.h"
#include "processmodel.h"


template< typename ApplOrderT,
typename TaskOrderT,
typename ThreadOrderT,
typename NodeOrderT >
class ProcessModelAppl
{

public:
ProcessModelAppl( ApplOrderT order = 0 ): traceGlobalOrder( order )
{}

~ProcessModelAppl()
{}

bool operator==( const ProcessModelAppl< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT >& other ) const
{
return traceGlobalOrder == other.traceGlobalOrder &&
tasks            == other.tasks;
}

size_t size() const { return tasks.size(); }
typename std::vector< ProcessModelTask< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > >::const_iterator cbegin() const { return tasks.cbegin(); };
typename std::vector< ProcessModelTask< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > >::const_iterator cend() const { return tasks.cend(); };

protected:
ApplOrderT traceGlobalOrder;
std::vector< ProcessModelTask< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > > tasks;

private:
template< typename ApplOrderU, typename TaskOrderU, typename ThreadOrderU, typename NodeOrderU > friend class ProcessModel;

};


