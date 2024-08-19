


#pragma once


#include <string>
#include <vector>

#include "paraverkerneltypes.h"

class KernelConnection;
class ProgressController;

class TraceShifter
{
public:
static TraceShifter *create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string shiftTimesFile,
TWindowLevel shiftLevel,
ProgressController *progress );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~TraceShifter()
{}

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) = 0;

virtual const std::vector< TTime > getShiftTimes() = 0;

private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};


class TraceShifterProxy : public TraceShifter
{
public:
virtual ~TraceShifterProxy();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

virtual const std::vector< TTime > getShiftTimes() override; 

private:
TraceShifter *myTraceShifter;
const KernelConnection *myKernel;

TraceShifterProxy( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string shiftTimesFile,
TWindowLevel shiftLevel,
ProgressController *progress );

friend TraceShifter *TraceShifter::create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string shiftTimesFile,
TWindowLevel shiftLevel,
ProgressController *progress );
};


