


#pragma once


#include <map>
#include "intervalhigh.h"
#include "semanticnotthread.h"

class KSingleWindow;
class SemanticNotThread;

class IntervalNotThread: public IntervalHigh
{
public:
IntervalNotThread()
{
function = nullptr;
}

IntervalNotThread( KTimeline *whichWindow, TWindowLevel whichLevel,
TObjectOrder whichOrder ):
IntervalHigh( whichLevel, whichOrder ), window( whichWindow )
{
function = nullptr;
}

virtual ~IntervalNotThread()
{
if ( begin != nullptr )
delete begin;
if ( end != nullptr )
delete end;
}

virtual KRecordList *init( TRecordTime initialTime, TCreateList create,
KRecordList *displayList = nullptr ) override;
virtual KRecordList *calcNext( KRecordList *displayList = nullptr, bool initCalc = false ) override;
virtual KRecordList *calcPrev( KRecordList *displayList = nullptr, bool initCalc = false ) override;

virtual KTimeline *getWindow() override
{
return ( KTimeline * ) window;
}

protected:
KTimeline *window;
SemanticNotThread *function;
TCreateList createList;

virtual void setChildren() override
{
if ( level == WORKLOAD )
{
if ( lastLevel != COMPOSEAPPLICATION )
{
lastLevel = COMPOSEAPPLICATION;
for ( TApplOrder i = 0; i < getWindowTrace()->totalApplications(); i++ )
{
childIntervals.push_back( getWindowInterval( COMPOSEAPPLICATION, i ) );
}
}
}
else if ( level == APPLICATION )
{
if ( lastLevel != COMPOSETASK )
{
lastLevel = COMPOSETASK;
for ( TTaskOrder i = getWindowTrace()->getFirstTask( order );
i <= getWindowTrace()->getLastTask( order ); i++ )
{
childIntervals.push_back( getWindowInterval( COMPOSETASK, i ) );
}
}
}
else if ( level == TASK )
{
if ( lastLevel != COMPOSETHREAD )
{
lastLevel = COMPOSETHREAD;
TApplOrder myAppl;
TTaskOrder myTask;
getWindowTrace()->getTaskLocation( order, myAppl, myTask );
for ( TThreadOrder i = getWindowTrace()->getFirstThread( myAppl, myTask );
i <= getWindowTrace()->getLastThread( myAppl, myTask ); i++ )
{
childIntervals.push_back( getWindowInterval( COMPOSETHREAD, i ) );
}
}
}
else if ( level == SYSTEM )
{
if ( lastLevel != COMPOSENODE )
{
lastLevel = COMPOSENODE;
for ( TNodeOrder i = 0; i < getWindowTrace()->totalNodes(); i++ )
{
childIntervals.push_back( getWindowInterval( COMPOSENODE, i ) );
}
}
}
else if ( level == NODE )
{
if ( lastLevel != COMPOSECPU )
{
lastLevel = COMPOSECPU;
for ( TCPUOrder i = getWindowTrace()->getFirstCPU( order );
i <= getWindowTrace()->getLastCPU( order ); i++ )
{
childIntervals.push_back( getWindowInterval( COMPOSECPU, i - 1 ) );
}
}
}
}

virtual KTrace *getWindowTrace() const override;
virtual TTraceLevel getWindowLevel() const override;
virtual Interval *getWindowInterval( TWindowLevel whichLevel, TObjectOrder whichOrder ) override;
virtual bool IsDerivedWindow() const override;
virtual TWindowLevel getComposeLevel( TTraceLevel whichLevel ) const override;

private:
SemanticHighInfo info;
std::multimap<TRecordTime,TObjectOrder> orderedChildren;
};



