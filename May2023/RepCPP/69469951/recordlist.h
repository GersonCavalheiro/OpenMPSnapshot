


#pragma once


#include <set>
#include "paraverkerneltypes.h"

class KernelConnection;

struct RLEvent
{
TEventType type;
TSemanticValue value;
};

struct RLComm
{
TObjectOrder partnerObject;
TRecordTime partnerTime;
TCommSize size;
TCommTag tag;
TCommID id;
};

struct RLRecord
{
public:
TRecordType getRecordType() const
{
return type;
}
TRecordTime getTime() const
{
return time;
}
TObjectOrder getOrder() const
{
return order;
}
TEventType getEventType() const
{
return UInfo.event.type;
}
TSemanticValue getEventValue() const
{
return UInfo.event.value;
}
TObjectOrder getCommPartnerObject() const
{
return UInfo.comm.partnerObject;
}
TRecordTime getCommPartnerTime() const
{
return UInfo.comm.partnerTime;
}
TCommSize getCommSize() const
{
return UInfo.comm.size;
}
TCommTag getCommTag() const
{
return UInfo.comm.tag;
}
TCommID getCommId() const
{
return UInfo.comm.id;
}
void setRecordType( TRecordType whichType )
{
type = whichType;
}
void setTime( TRecordTime whichTime )
{
time = whichTime;
}
void setOrder( TObjectOrder whichOrder )
{
order = whichOrder;
}
void setEventType( TEventType whichType )
{
UInfo.event.type = whichType;
}
void setEventValue( TSemanticValue whichValue )
{
UInfo.event.value = whichValue;
}
void setCommPartnerObject( TObjectOrder whichOrder )
{
UInfo.comm.partnerObject = whichOrder;
}
void setCommPartnerTime( TRecordTime whichTime )
{
UInfo.comm.partnerTime = whichTime;
}
void setCommSize( TCommSize whichSize )
{
UInfo.comm.size = whichSize;
}
void setCommTag( TCommTag whichTag )
{
UInfo.comm.tag = whichTag;
}
void setCommId( TCommID whichID )
{
UInfo.comm.id = whichID;
}
private:
TRecordType type;
TRecordTime time;
TObjectOrder order;
union
{
RLEvent event;
RLComm comm;
} UInfo;
};

struct ltrecord
{
bool operator()( const RLRecord& r1, const RLRecord& r2 ) const
{
if ( r1.getTime() < r2.getTime() )
return true;

return false;
}
};


class RecordList
{
public:
typedef std::multiset<RLRecord, ltrecord>::iterator iterator;

static RecordList *create( RecordList *whichList );

virtual ~RecordList() {}

virtual void clear() = 0;
virtual void erase( iterator first, iterator last ) = 0;
virtual RecordList::iterator begin() = 0;
virtual RecordList::iterator end() = 0;
virtual bool newRecords() const = 0;
virtual RecordList *clone()
{
return nullptr;
};
};

class RecordListProxy: public RecordList
{
public:
virtual ~RecordListProxy() {};

virtual void clear() override;
virtual void erase( iterator first, iterator last ) override;
virtual RecordList::iterator begin() override;
virtual RecordList::iterator end() override;
virtual bool newRecords() const override;
virtual RecordList *clone() override;

private:
RecordListProxy( RecordList *whichList );

RecordList *myRecordList;

friend RecordList *RecordList::create( RecordList * );
};


