

#pragma once

#include <string>

#include "tracebodyio.h"


constexpr bool prv_atoll_v( std::string::const_iterator& it, const std::string::const_iterator& end )
{
return true;
}

template <typename T, typename... Targs>
constexpr bool prv_atoll_v( std::string::const_iterator& it, const std::string::const_iterator& end, T& result, Targs&... Fargs )
{
result = 0;
int negative = 1;

if( it == end )
return false;

if( *it == '-' )
{
if( std::is_unsigned<T>::value )
return false;
negative = -1;
++it;
}

if( *it >= '0' && *it <= '9' )
{
result = ( *it++ - '0' );
while( *it >= '0' && *it <= '9' )
result = ( result * 10 ) + ( *it++ - '0' );

result *= negative;
}

if( it == end )
return sizeof...( Targs ) == 0;

return prv_atoll_v( ++it, end, Fargs... );
}


template< class    TraceStreamT,
class    RecordContainerT,
class    ProcessModelT,
class    ResourceModelT,
typename StateT,
typename EventTypeT,
class    MetadataManagerT,
typename RecordTimeT,
class    RecordT>
class TraceBodyIO_v1 : public TraceBodyIO<TraceStreamT,
RecordContainerT,
ProcessModelT,
ResourceModelT,
StateT,
EventTypeT,
MetadataManagerT,
RecordTimeT,
RecordT>
{
public:
TraceBodyIO_v1() {}

static constexpr char CommentRecord    = '#';
static constexpr char StateRecord      = '1';
static constexpr char EventRecord      = '2';
static constexpr char CommRecord       = '3';
static constexpr char GlobalCommRecord = '4';

bool ordered() const override;
void read( TraceStreamT& file,
RecordContainerT& records,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
std::unordered_set<StateT>& states,
std::unordered_set<EventTypeT>& events,
MetadataManagerT& traceInfo,
RecordTimeT& endTime ) const override;
void write( std::fstream& whichStream,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
RecordT *record ) const override;
bool writePendingMultiEvent( const ProcessModelT& whichProcessModel ) const;

protected:

private:
typedef struct TMultiEventCommonInfo
{
std::fstream *myStream;

TThreadOrder thread;
TCPUOrder cpu;
TRecordTime time;
}
TMultiEventCommonInfo;

static TMultiEventCommonInfo multiEventCommonInfo;
static std::string multiEventLine;

static std::string line;
static std::ostringstream ostr;

bool validRecordLocation( const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
TCPUOrder whichCPU,
TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) const;

void readTraceInfo( const std::string& line, MetadataManagerT& traceInfo ) const;

void readState( const std::string& line,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
RecordContainerT& records,
std::unordered_set<StateT>& states ) const;
void readEvent( const std::string& line,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
RecordContainerT& records,
std::unordered_set<EventTypeT>& events ) const;
void readComm( const std::string& line,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
RecordContainerT& records ) const;
void readGlobalComm( const std::string& line, RecordContainerT& records ) const;
bool readCommon( const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
std::string::const_iterator& it,
const std::string::const_iterator& end,
TCPUOrder& CPU,
TApplOrder& appl,
TTaskOrder& task,
TThreadOrder& thread,
RecordTimeT& time ) const;

void bufferWrite( std::fstream& whichStream, bool writeReady, bool lineClear = true  ) const;

bool writeState( const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
const RecordT *record ) const;
void appendEvent( const RecordT *record ) const;
bool writeEvent( const ProcessModelT& whichProcessModel,
const RecordT *record,
bool needCommons = true ) const;
bool writeComm( const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
const RecordT *record ) const;
bool writeGlobalComm( const ProcessModelT& whichProcessModel,
const RecordT *record ) const;
void writeCommon( std::ostringstream& line,
const ProcessModelT& whichProcessModel,
const ResourceModelT& whichResourceModel,
const RecordT *record ) const;

bool sameMultiEvent( const RecordT *record ) const;
};

#include "tracebodyio_v1.cpp"

