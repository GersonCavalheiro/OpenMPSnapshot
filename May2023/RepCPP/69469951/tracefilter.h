


#pragma once


#include <map>

#include "localkernel.h"

class TraceOptions;
class KernelConnection;
class ProgressController;

class TraceFilter
{
public:
static TraceFilter *create( const KernelConnection *whichKernel,
char *traceIn,
char *traceOut,
TraceOptions *options,
ProgressController *progress,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~TraceFilter()
{}

private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};

class TraceFilterProxy : public TraceFilter
{
public:
virtual ~TraceFilterProxy();

private:
TraceFilter *myTraceFilter;

TraceFilterProxy( const KernelConnection *whichKernel,
char *traceIn,
char *traceOut,
TraceOptions *options,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable,
ProgressController *progress );

friend TraceFilter *TraceFilter::create( const KernelConnection *kernelConnection,
char *traceIn,
char *traceOut,
TraceOptions *options,
ProgressController *progress,
const std::map< TTypeValuePair, TTypeValuePair >& whichTranslationTable );
};



