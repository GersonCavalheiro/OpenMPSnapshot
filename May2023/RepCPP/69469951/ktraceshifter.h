


#pragma once


#include <string>
#include <vector>

#include "traceshifter.h"
#include "ktraceeditsequence.h"
#include "paraverkerneltypes.h"
#include "progresscontroller.h"


class KTraceShifter : public TraceShifter
{
public:
KTraceShifter( const KernelConnection *myKernel,
std::string traceIn,
std::string traceOut,
std::string whichShiftTimes,
TWindowLevel shiftLevel,
ProgressController *progress = nullptr );
~KTraceShifter();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

virtual const std::vector< TTime > getShiftTimes() override { return shiftTimes; }

virtual const TTime getMaxShiftTime() { return maxShiftTime; }

private:
TraceEditSequence *mySequence;

std::vector<std::string> traces;

std::vector< TTime > shiftTimes;
std::vector< TTime > readShiftTimes( std::string shiftTimesFile );
TTime maxShiftTime;
};


