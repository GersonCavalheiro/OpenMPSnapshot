


#pragma once


#include <string>
#include "paraverkerneltypes.h"

class KernelConnection;
class ProgressController;

class EventDrivenCutter
{
public:
static EventDrivenCutter *create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~EventDrivenCutter()
{}

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) = 0;

private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};


class EventDrivenCutterProxy : public EventDrivenCutter
{
public:
virtual ~EventDrivenCutterProxy();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

private:
EventDrivenCutter *myEventDrivenCutter;
const KernelConnection *myKernel;

EventDrivenCutterProxy( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress );

friend EventDrivenCutter *EventDrivenCutter::create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress );
};



