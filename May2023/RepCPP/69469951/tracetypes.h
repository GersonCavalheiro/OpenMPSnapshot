

#pragma once

#include <cstdint>

using TTraceSize = uint64_t;

#ifdef EXTENDED_OBJECTS_ENABLED
using TObjectOrder = uint32_t;
#else
using TObjectOrder = uint16_t;
#endif

using TThreadOrder = TObjectOrder;
using TTaskOrder   = TObjectOrder;
using TApplOrder   = TObjectOrder;

using TNodeOrder   = TObjectOrder;
using TCPUOrder    = TObjectOrder;

using TSemanticValue = double;

using TTime       = double;
using TRecordTime = TTime;
using TTimeUnit   = uint16_t;

constexpr TTimeUnit NS   = 0;
constexpr TTimeUnit US   = 1;
constexpr TTimeUnit MS   = 2;
constexpr TTimeUnit SEC  = 3;
constexpr TTimeUnit MIN  = 4;
constexpr TTimeUnit HOUR = 5;
constexpr TTimeUnit DAY  = 6;

using TRecordType = uint16_t;

constexpr TRecordType BEGIN    = 0x0001;
constexpr TRecordType END      = 0x0002;
constexpr TRecordType STATE    = 0x0004;
constexpr TRecordType EVENT    = 0x0008;
constexpr TRecordType LOG      = 0x0010;
constexpr TRecordType PHY      = 0x0020;
constexpr TRecordType SEND     = 0x0040;
constexpr TRecordType RECV     = 0x0080;
constexpr TRecordType COMM     = 0x0100;
constexpr TRecordType GLOBCOMM = 0x0200;
constexpr TRecordType RRECV    = 0x0400;
constexpr TRecordType RSEND    = 0x0800;

constexpr TRecordType EMPTYREC = STATE + EVENT + COMM;

using TCommID     = uint64_t;
using TCommSize   = int64_t;
using TCommTag    = int64_t;
using TEventType  = uint32_t;
using TEventValue = int64_t;
using TState      = uint32_t;

enum class TTraceLevel
{
NONE = 0,
WORKLOAD, APPLICATION, TASK, THREAD,
SYSTEM, NODE, CPU
};

constexpr TTraceLevel& operator++( TTraceLevel& whichLevel )
{
return whichLevel = static_cast<TTraceLevel>( static_cast<size_t>( whichLevel ) + 1 );
}

constexpr TTraceLevel operator++( TTraceLevel& whichLevel, int )
{
TTraceLevel tmp = whichLevel;
whichLevel = static_cast<TTraceLevel>( static_cast<size_t>( whichLevel ) + 1 );
return tmp;
}

using ParaverColor = unsigned char;
