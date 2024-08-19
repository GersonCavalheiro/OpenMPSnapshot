


#pragma once

#include <vector>
#include "paraverkerneltypes.h"

class MemoryBlocks;
class Trace;

typedef struct {} TData;

class MemoryTrace
{
public:
class iterator
{
public:
iterator();
iterator( const Trace *whichTrace );
virtual ~iterator() = default;

virtual void operator++() = 0;
virtual void operator--() = 0;
virtual MemoryTrace::iterator& operator=( const MemoryTrace::iterator& copy );

virtual bool operator==( const iterator &it ) const;
virtual bool operator!=( const iterator &it ) const;
virtual bool isNull() const;

virtual iterator *clone() const = 0;

virtual TRecordType    getRecordType() const = 0;
virtual TRecordTime    getTime() const = 0;
virtual TThreadOrder   getThread() const = 0;
virtual TCPUOrder      getCPU() const = 0;
virtual TObjectOrder   getOrder() const = 0;
virtual TEventType     getEventType() const = 0;
virtual TSemanticValue getEventValue() const = 0;
virtual TEventValue    getEventValueAsIs() const = 0;
virtual TState         getState() const = 0;
virtual TRecordTime    getStateEndTime() const = 0;
virtual TCommID        getCommIndex() const = 0;

virtual TThreadOrder getSenderThread() const;
virtual TCPUOrder    getSenderCPU() const;
virtual TThreadOrder getReceiverThread() const;
virtual TCPUOrder    getReceiverCPU() const;
virtual TCommTag     getCommTag() const;
virtual TCommSize    getCommSize() const;
virtual TRecordTime  getLogicalSend() const;
virtual TRecordTime  getLogicalReceive() const;
virtual TRecordTime  getPhysicalSend() const;
virtual TRecordTime  getPhysicalReceive() const;

virtual void         setTime( const TRecordTime time ) = 0;
virtual void         setRecordType( const TRecordType whichType ) = 0;
virtual void         setStateEndTime( const TRecordTime whichEndTime ) = 0;

virtual TData *getRecord() const
{
return record;
}
virtual void setRecord( TData *whichRecord )
{
record = whichRecord;
}

protected :
TData *record = nullptr;
const Trace *myTrace;
};

MemoryTrace()
{}

virtual ~MemoryTrace()
{}

virtual void insert( MemoryBlocks *blocks ) = 0;
virtual TTime finish( TTime headerTime, Trace *whichTrace ) = 0;

virtual MemoryTrace::iterator* empty() const = 0;
virtual MemoryTrace::iterator* begin() const = 0;
virtual MemoryTrace::iterator* end() const = 0;

virtual MemoryTrace::iterator* threadBegin( TThreadOrder whichThread ) const = 0;
virtual MemoryTrace::iterator* threadEnd( TThreadOrder whichThread ) const = 0;
virtual MemoryTrace::iterator* CPUBegin( TCPUOrder whichCPU ) const = 0;
virtual MemoryTrace::iterator* CPUEnd( TCPUOrder whichCPU ) const = 0;


virtual void getRecordByTimeThread( std::vector<MemoryTrace::iterator *>& listIter,
TRecordTime whichTime ) const = 0;
virtual void getRecordByTimeCPU( std::vector<MemoryTrace::iterator *>& listIter,
TRecordTime whichTime ) const = 0;

protected:

private:

};




