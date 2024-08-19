


#pragma once


#include <stddef.h>
#include <vector>
#include <string>
#include "prvtypes.h"
#include "config_traits.h"
#include "utils/traceparser/tracetypes.h"

static const TRecordTime factorTable[ DAY + 1 ] =
{
1,   
1E3, 
1E3, 
1E3, 
60,  
60,  
24   
};


enum TWindowLevel
{
NONE = static_cast<int>( TTraceLevel::NONE ),
WORKLOAD = static_cast<int>( TTraceLevel::WORKLOAD ),
APPLICATION = static_cast<int>( TTraceLevel::APPLICATION ),
TASK = static_cast<int>( TTraceLevel::TASK ),
THREAD = static_cast<int>( TTraceLevel::THREAD ),
SYSTEM = static_cast<int>( TTraceLevel::SYSTEM ),
NODE = static_cast<int>( TTraceLevel::NODE ),
CPU = static_cast<int>( TTraceLevel::CPU ),
TOPCOMPOSE1, TOPCOMPOSE2, COMPOSEWORKLOAD, COMPOSEAPPLICATION, COMPOSETASK,
COMPOSETHREAD, COMPOSESYSTEM, COMPOSENODE, COMPOSECPU,
DERIVED,
EXTRATOPCOMPOSE1
};

typedef PRV_UINT16 TFilterNumParam;

typedef PRV_UINT16 TParamIndex;
typedef std::vector<double> TParamValue;

typedef PRV_UINT16 TCreateList;
static const TCreateList NOCREATE = 0x00;
static const TCreateList CREATEEVENTS = 0x01;
static const TCreateList CREATECOMMS = 0x02;

typedef double     THistogramLimit;
typedef PRV_UINT32 THistogramColumn;

enum THistoTotals
{
TOTAL = 0, AVERAGE, MAXIMUM, MINIMUM, STDEV, AVGDIVMAX, NUMTOTALS
};

enum class THistoSortCriteria
{
TOTAL = 0, AVERAGE, MAXIMUM, MINIMUM, STDEV, AVGDIVMAX, CUSTOM
};

enum SemanticInfoType
{
NO_TYPE = 0, 
SAME_TYPE, 
OBJECT_TYPE, 
APPL_TYPE,
TASK_TYPE,
THREAD_TYPE,
NODE_TYPE,
CPU_TYPE,
TIME_TYPE,
STATE_TYPE,
EVENTTYPE_TYPE,
EVENTVALUE_TYPE,
COMMSIZE_TYPE,
COMMTAG_TYPE,
BANDWIDTH_TYPE
};

enum TSemanticGroup
{
COMPOSE_GROUP = 0,
DERIVED_GROUP, CPU_GROUP, NOTTHREAD_GROUP, STATE_GROUP,
EVENT_GROUP, COMM_GROUP, OBJECT_GROUP
};

static const std::string        GZIPPED_PRV_SUFFIX = ".prv.gz";
static const std::string                PRV_SUFFIX = ".prv";
static const std::string                CFG_SUFFIX = ".cfg";
static const std::string        DIMEMAS_CFG_SUFFIX = ".cfg";
static const std::string                PCF_SUFFIX = ".pcf";
static const std::string                ROW_SUFFIX = ".row";
static const std::string TRACE_TOOL_OPTIONS_SUFFIX = ".xml";
static const std::string               OTF2_SUFFIX = ".otf2";

static const std::string FILTER_SEP = ".";

static const std::string  BMP_SUFFIX = ".bmp";
static const std::string JPEG_SUFFIX = ".jpg";
static const std::string  PNG_SUFFIX = ".png";
static const std::string  XPM_SUFFIX = ".xpm";



constexpr TState IDLE = 0;
constexpr TState RUNNING = 1;
