


#pragma once


#include <string>
#include <vector>
#include "traceeditsequence.h"

using std::vector;


enum class TTraceEditActionType
{
TraceToTrace = 0,
TraceToRecord,
RecordToTrace,
RecordToRecord
};

class TraceEditAction
{
public:
TraceEditAction( TraceEditSequence *whichSequence ) : mySequence( whichSequence ) {}
virtual ~TraceEditAction() {}

virtual TTraceEditActionType getType() const = 0;
virtual vector<TSequenceStates> getStateDependencies() const = 0;

protected:
TraceEditSequence *mySequence;

private:

};


class TraceToTraceAction: public TraceEditAction
{
public:
TraceToTraceAction( TraceEditSequence *whichSequence ) : TraceEditAction( whichSequence ) {}
virtual ~TraceToTraceAction() {}

virtual TTraceEditActionType getType() const override
{
return TTraceEditActionType::TraceToTrace;
}

virtual bool execute( std::string whichTrace ) = 0;

protected:

private:

};





