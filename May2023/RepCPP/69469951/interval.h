


#pragma once


#include <set>
#include "memorytrace.h"
#include "krecordlist.h"

class KTimeline;

class Interval
{
public:
Interval()
{
begin = nullptr;
end = nullptr;
currentValue = 0.0;
notWindowInits = false;
}

Interval( TWindowLevel whichLevel, TObjectOrder whichOrder ):
order( whichOrder )
{
begin = nullptr;
end = nullptr;
currentValue = 0.0;
notWindowInits = false;
}

virtual ~Interval()
{
myDisplayList.clear();
}

virtual TRecordTime getBeginTime() const
{
return begin->getTime();
}

virtual TRecordTime getEndTime() const
{
return end->getTime();
}

virtual TSemanticValue getValue() const
{
return currentValue;
}

virtual MemoryTrace::iterator *getBegin() const
{
return begin;
}

virtual MemoryTrace::iterator *getEnd() const
{
return end;
}

virtual MemoryTrace::iterator *getTraceEnd() const = 0;

virtual TWindowLevel getLevel() const = 0;

TObjectOrder getOrder()
{
return order;
}

KRecordList *getRecordList()
{
return &myDisplayList;
}

bool getNotWindowInits() const
{
return notWindowInits;
}

void setNotWindowInits( bool whichValue )
{
notWindowInits = whichValue;
}

virtual KRecordList *init( TRecordTime initialTime, TCreateList create,
KRecordList *displayList = nullptr ) = 0;
virtual KRecordList *calcNext( KRecordList *displayList = nullptr, bool initCalc = false ) = 0;
virtual KRecordList *calcPrev( KRecordList *displayList = nullptr, bool initCalc = false ) = 0;

virtual KTimeline *getWindow() = 0;

protected:
TObjectOrder order;
MemoryTrace::iterator *begin;
MemoryTrace::iterator *end;
TSemanticValue currentValue;
KRecordList myDisplayList;
bool notWindowInits;

private:

};



