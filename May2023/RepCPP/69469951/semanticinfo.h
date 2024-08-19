


#pragma once


#include <vector>
#include "paraverkerneltypes.h"
#include "interval.h"
#include "memorytrace.h"

struct SemanticInfo
{
Interval *callingInterval;
};


struct SemanticThreadInfo: public SemanticInfo
{
MemoryTrace::iterator *it;
};


struct SemanticHighInfo: public SemanticInfo
{
std::vector<TSemanticValue> values;
TObjectOrder lastChanged;
TRecordTime dataBeginTime;
TRecordTime dataEndTime;
bool newControlBurst;
};


