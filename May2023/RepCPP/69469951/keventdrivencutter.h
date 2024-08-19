


#pragma once


#include <string>
#include <vector>

#include "eventdrivencutter.h"
#include "ktraceeditsequence.h"

class ProgressController;
class KernelConnection;

class KEventDrivenCutter : public EventDrivenCutter
{
public:
KEventDrivenCutter( const KernelConnection *myKernel,
std::string traceIn,
std::string traceOut,
TEventType whichEvent,
ProgressController *progress = nullptr );
~KEventDrivenCutter();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

private:
TraceEditSequence *mySequence;

std::vector<std::string> traces;

};


