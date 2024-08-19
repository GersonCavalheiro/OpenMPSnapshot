


#pragma once


#include "intervalhigh.h"
#include "intervalshift.h"
#include "semanticderived.h"

class KTimeline;
class KDerivedWindow;
class SemanticDerived;

class IntervalDerived: public IntervalHigh
{
public:
IntervalDerived();
IntervalDerived( KDerivedWindow *whichWindow,
TWindowLevel whichLevel,
TObjectOrder whichOrder );

virtual ~IntervalDerived();

virtual KRecordList *init( TRecordTime initialTime, TCreateList create,
KRecordList *displayList = nullptr ) override;
virtual KRecordList *calcNext( KRecordList *displayList = nullptr, bool initCalc = false ) override;
virtual KRecordList *calcPrev( KRecordList *displayList = nullptr, bool initCalc = false ) override;

virtual KTimeline *getWindow() override
{
return ( KTimeline * ) window;
}

protected:
KDerivedWindow *window;
SemanticDerived *function;
TCreateList createList;

virtual void setChildren() override;

virtual KTrace *getWindowTrace() const override;
virtual TTraceLevel getWindowLevel() const override;
virtual Interval *getWindowInterval( TWindowLevel whichLevel, TObjectOrder whichOrder ) override;
virtual bool IsDerivedWindow() const override;
virtual TWindowLevel getComposeLevel( TTraceLevel whichLevel ) const override;

private:
SemanticHighInfo info;

IntervalShift shift1;
IntervalShift shift2;

};


