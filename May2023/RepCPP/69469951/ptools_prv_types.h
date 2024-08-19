


#pragma once


#include "ptools_types.h"

typedef char          prvMagic_t[8];
typedef char          prvDate_t[23];
typedef PTOOLS_UINT64 prvTime_t;
typedef char          prvTimeUnit_t[2];

typedef PTOOLS_UINT32 prvObjectID_t;

typedef prvObjectID_t prvNodeID_t;
typedef prvObjectID_t prvCPUID_t;

typedef prvObjectID_t prvNumNodes_t;
typedef prvObjectID_t prvNumCPUs_t;

typedef prvObjectID_t prvApplicationID_t;
typedef prvObjectID_t prvTaskID_t;
typedef prvObjectID_t prvThreadID_t;

typedef prvObjectID_t prvNumApplications_t;
typedef prvObjectID_t prvNumTasks_t;
typedef prvObjectID_t prvNumThreads_t;

typedef PTOOLS_UINT32 prvCommunicatorID_t;

typedef PTOOLS_UINT32 prvNumCommunicators_t;

typedef char          prvRecordType_t;

typedef PTOOLS_UINT16 prvState_t;

typedef PTOOLS_UINT32 prvEventType_t;
typedef PTOOLS_INT64  prvEventValue_t;

typedef PTOOLS_UINT32 prvCommSize_t;
typedef PTOOLS_UINT32 prvCommTag_t;



