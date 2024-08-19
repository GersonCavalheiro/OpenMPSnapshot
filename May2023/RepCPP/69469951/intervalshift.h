


#pragma once


#include <deque>
#include "intervalhigh.h"

using std::deque;

class KTimeline;
class KDerivedWindow;

class IntervalShift : public IntervalHigh
{
public:
IntervalShift()
{
semanticShift = 0;
}

IntervalShift( KDerivedWindow *whichWindow, TWindowLevel whichLevel,
TObjectOrder whichOrder ):
IntervalHigh( whichLevel, whichOrder ), window( whichWindow )
{
semanticShift = 0;
}

virtual ~IntervalShift()
{
if ( begin != nullptr )
delete begin;
if ( end != nullptr )
delete end;

clearSemanticBuffer();
}

IntervalShift *clone() const;

virtual KRecordList *init( TRecordTime initialTime, TCreateList create,
KRecordList *displayList = nullptr ) override;
virtual KRecordList *calcNext( KRecordList *displayList = nullptr, bool initCalc = false ) override;
virtual KRecordList *calcPrev( KRecordList *displayList = nullptr, bool initCalc = false ) override;

virtual KTimeline *getWindow() override
{
return ( KTimeline * ) window;
}

void setChildInterval( Interval *whichInterval );
void setSemanticShift( PRV_INT16 whichShift );


protected:
virtual void setChildren() override;

virtual KTrace *getWindowTrace() const override;
virtual TTraceLevel getWindowLevel() const override;
virtual Interval *getWindowInterval( TWindowLevel whichLevel, TObjectOrder whichOrder ) override;
virtual bool IsDerivedWindow() const override;
virtual TWindowLevel getComposeLevel( TTraceLevel whichLevel ) const override;

virtual TRecordTime getBeginTime() const override;
virtual TRecordTime getEndTime() const override;
virtual TSemanticValue getValue() const override;
virtual MemoryTrace::iterator *getBegin() const override;
virtual MemoryTrace::iterator *getEnd() const override;

private:
class ShiftSemanticInfo
{
public:
ShiftSemanticInfo() : semanticValue( 0.0 ), begin( nullptr ), end( nullptr )
{}

ShiftSemanticInfo( TSemanticValue whichSemanticValue,
MemoryTrace::iterator *whichBegin,
MemoryTrace::iterator *whichEnd ) :
semanticValue( whichSemanticValue ), begin( whichBegin ), end( whichEnd )
{}

~ShiftSemanticInfo()
{
if( begin != nullptr ) delete begin;
if( end != nullptr ) delete end;
}

TSemanticValue semanticValue;
MemoryTrace::iterator *begin;
MemoryTrace::iterator *end;
};

KDerivedWindow *window;

std::deque<IntervalShift::ShiftSemanticInfo> semanticBuffer;

PRV_INT16 semanticShift;
PRV_UINT16 bufferSize;

void popSemanticBuffer();
void clearSemanticBuffer();
void addSemanticBuffer();

};

