

#pragma once

#include "tracetypes.h"

constexpr char LEVEL_WORKLOAD[] =     "WORKLOAD";
constexpr char LEVEL_APPLICATION[] =  "APPL";
constexpr char LEVEL_TASK[] =         "TASK";
constexpr char LEVEL_THREAD[] =       "THREAD";
constexpr char LEVEL_SYSTEM[] =       "SYSTEM";
constexpr char LEVEL_NODE[] =         "NODE";
constexpr char LEVEL_CPU[] =          "CPU";

static const std::string LABEL_LEVELS[ static_cast<int>( TTraceLevel::CPU ) + 1 ] =
{
"NONE",
LEVEL_WORKLOAD,
LEVEL_APPLICATION,
LEVEL_TASK,
LEVEL_THREAD,
LEVEL_SYSTEM,
LEVEL_NODE,
LEVEL_CPU
};
