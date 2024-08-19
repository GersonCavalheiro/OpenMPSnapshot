

#pragma once

#include "utils/traceparser/tracebodyio.h"
#include "tracestream.h"
#include "ParaverMetadataManager.h"
#include "memorytrace.h"
#include "utils/traceparser/processmodel.h"
#include "utils/traceparser/resourcemodel.h"
#include "memoryblocks.h"


#define PARAM_TRACEBODY_CLASS TraceStream, \
MemoryBlocks, \
ProcessModel<>, \
ResourceModel<>, \
TState, \
TEventType, \
MetadataManager, \
TRecordTime, \
MemoryTrace::iterator

class TraceBodyIOFactory
{
public:
static TraceBodyIO< PARAM_TRACEBODY_CLASS > *createTraceBody( TraceStream *file, Trace *trace, ProcessModel<>& whichProcessModel );
static TraceBodyIO< PARAM_TRACEBODY_CLASS > *createTraceBody();
};
