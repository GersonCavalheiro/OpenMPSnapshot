


#pragma once


#include <set>
#include "recordlist.h"
#include "memorytrace.h"

class KTimeline;

class KRecordList: public RecordList
{
public:
KRecordList();
virtual ~KRecordList();

virtual void clear();
virtual void erase( iterator first, iterator last );
virtual iterator begin();
virtual iterator end();
virtual bool newRecords() const;

virtual void insert( KTimeline *window, MemoryTrace::iterator *it );
virtual RecordList *clone();

protected:

private:
std::multiset<RLRecord, ltrecord> list;
bool newRec;
};



