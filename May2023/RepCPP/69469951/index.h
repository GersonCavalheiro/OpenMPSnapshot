


#pragma once


#include "paraverkerneltypes.h"

#include <map>

static PRV_UINT32 INDEX_STEP = 10000;

template <typename RecordType>
class Index
{
public:
Index( PRV_UINT32 step = INDEX_STEP );
~Index();

void indexRecord( TRecordTime time, RecordType rec );
bool findRecord( TRecordTime time, RecordType& record ) const;

private:
typedef std::map< TRecordTime, RecordType > TTraceIndex;

PRV_UINT32 indexStep;
TTraceIndex baseIndex;
PRV_UINT32 counter;
};

#include "index_impl.h"


