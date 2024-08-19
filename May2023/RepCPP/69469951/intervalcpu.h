


#pragma once


#include <unordered_map>

#include "intervalhigh.h"
#include "semanticcpu.h"
#include "semanticthread.h"
#include "semanticcompose.h"
#include "intervalcompose.h"
#include "intervalthread.h"

class KSingleWindow;
class SemanticCPU;

class IntervalCPU: public IntervalHigh
{
public:
IntervalCPU()
{
begin = nullptr;
end = nullptr;
function = nullptr;
functionThread = nullptr;
functionComposeThread = nullptr;
}

IntervalCPU( KSingleWindow *whichWindow, TWindowLevel whichLevel, TObjectOrder whichOrder );

virtual ~IntervalCPU()
{
if ( begin != nullptr )
delete begin;
if ( end != nullptr )
delete end;
if( functionThread != nullptr )
delete functionThread;
if( functionComposeThread != nullptr )
delete functionComposeThread;
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

MemoryTrace::iterator *getTraceEnd() const override;

protected:
KSingleWindow *window;
SemanticCPU *function;
TCreateList createList;
SemanticThread *functionThread;
SemanticCompose *functionComposeThread;
std::unordered_map<TThreadOrder, IntervalCompose *> intervalCompose;
std::unordered_map<TThreadOrder, IntervalThread *> intervalThread;
TRecordTime currentInitialTime;
Plain::TRecord virtualRecord;

private:
virtual MemoryTrace::iterator *getPrevRecord( MemoryTrace::iterator *it,
KRecordList *displayList );
virtual MemoryTrace::iterator *getNextRecord( MemoryTrace::iterator *it,
KRecordList *displayList );

virtual void setChildren() override {};
virtual KTrace *getWindowTrace() const override;
virtual TTraceLevel getWindowLevel() const override;
virtual Interval *getWindowInterval( TWindowLevel whichLevel,
TObjectOrder whichOrder ) override;
virtual bool IsDerivedWindow() const override;
virtual TWindowLevel getComposeLevel( TTraceLevel whichLevel ) const override;

};



