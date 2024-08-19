


#pragma once


#include "interval.h"
#include "ktrace.h"

class IntervalHigh: public Interval
{
public:
IntervalHigh()
{}

IntervalHigh( TWindowLevel whichLevel, TObjectOrder whichOrder ):
Interval( whichLevel, whichOrder ), level( whichLevel ), lastLevel( NONE )
{}

~IntervalHigh()
{}

TWindowLevel getLevel() const override
{
return level;
}

virtual MemoryTrace::iterator *getTraceEnd() const override
{
return childIntervals[ 0 ]->getTraceEnd();
}

protected:
std::vector<Interval *> childIntervals;

TWindowLevel level;
TWindowLevel lastLevel;

virtual void setChildren() = 0;

virtual TTraceLevel getWindowLevel() const = 0;
virtual Interval *getWindowInterval( TWindowLevel whichLevel, TObjectOrder whichOrder ) = 0;
virtual bool IsDerivedWindow() const = 0;
virtual TWindowLevel getComposeLevel( TTraceLevel whichLevel ) const = 0;
virtual KTrace *getWindowTrace() const = 0;

private:

};



