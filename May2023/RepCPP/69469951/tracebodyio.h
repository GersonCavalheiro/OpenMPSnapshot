

#pragma once

#include <unordered_set>
#include <fstream>

template< class    TraceStreamT,
class    RecordContainerT,
class    ProcessModelT,
class    ResourceModelT,
typename StateT,
typename EventTypeT,
class    MetadataManagerT,
typename RecordTimeT,
class    RecordT>
class TraceBodyIO
{
public:
TraceBodyIO() = default;
virtual ~TraceBodyIO() = default;

virtual bool ordered() const = 0;
virtual void read( TraceStreamT& file,
RecordContainerT& records,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
std::unordered_set<StateT>& states,
std::unordered_set<EventTypeT>& events,
MetadataManagerT& traceInfo,
RecordTimeT& endTime ) const = 0;
virtual void write( std::fstream& whichStream,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
RecordT *record ) const = 0;

protected:

};



