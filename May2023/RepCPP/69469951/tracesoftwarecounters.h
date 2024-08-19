


#pragma once


#include <string>

class TraceOptions;
class KernelConnection;
class ProgressController;

class TraceSoftwareCounters
{
public:
static TraceSoftwareCounters *create( KernelConnection *whichKernel,
char *traceIn,
char *traceOut,
TraceOptions *options,
ProgressController *progress );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~TraceSoftwareCounters()
{}

private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};

class TraceSoftwareCountersProxy : public TraceSoftwareCounters
{
public:
virtual ~TraceSoftwareCountersProxy();

private:
TraceSoftwareCounters *myTraceSoftwareCounters;

TraceSoftwareCountersProxy( KernelConnection *whichKernel,
char *traceIn,
char *traceOut,
TraceOptions *options,
ProgressController *progress );

friend TraceSoftwareCounters *TraceSoftwareCounters::create( KernelConnection *kernelConnection,
char *traceIn,
char *traceOut,
TraceOptions *options,
ProgressController *progress );
};




