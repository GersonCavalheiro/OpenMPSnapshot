


#pragma once


#include "interval.h"
#include "semanticthread.h"

class KSingleWindow;
class SemanticThread;

class IntervalThread: public Interval
{
public:
IntervalThread()
{
begin = nullptr;
end = nullptr;
function = nullptr;
}

IntervalThread( KSingleWindow *whichWindow, TWindowLevel whichLevel,
TObjectOrder whichOrder ):
Interval( whichLevel, whichOrder ), window( whichWindow )
{
function = nullptr;
}

virtual ~IntervalThread()
{
if ( begin != nullptr )
delete begin;
if ( end != nullptr )
delete end;
}

virtual KRecordList *init( TRecordTime initialTime,
TCreateList create,
KRecordList *displayList = nullptr ) override;
virtual KRecordList *calcNext( KRecordList *displayList = nullptr, bool initCalc = false ) override;
virtual KRecordList *calcPrev( KRecordList *displayList = nullptr, bool initCalc = false ) override;

virtual KTimeline *getWindow() override
{
return ( KTimeline * ) window;
}

virtual void setSemanticFunction( SemanticThread *whichFunction )
{
function = whichFunction;
}

virtual TWindowLevel getLevel() const override
{
return THREAD;
}

MemoryTrace::iterator *getTraceEnd() const override;

protected:
KSingleWindow *window;
SemanticThread *function;
TCreateList createList;

private:
virtual MemoryTrace::iterator *getNextRecord( MemoryTrace::iterator *it,
KRecordList *displayList );
virtual MemoryTrace::iterator *getPrevRecord( MemoryTrace::iterator *it,
KRecordList *displayList );

};



