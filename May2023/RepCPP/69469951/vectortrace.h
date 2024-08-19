

#pragma once


#include "memorytrace.h"
#include "vectortracedefines.h"

class Trace;
class VectorBlocks;

class VectorTrace : public MemoryTrace
{
public:
class iterator : public MemoryTrace::iterator
{
public:
iterator( TThreadRecordContainer::iterator whichRecord, const Trace *whichTrace, VectorBlocks *whichBlocks, TThreadOrder whichThread );
virtual void operator++() override;
virtual void operator--() override;
virtual MemoryTrace::iterator& operator=( const MemoryTrace::iterator& copy ) override;

virtual bool operator==( const MemoryTrace::iterator &it ) const override;
virtual bool operator!=( const MemoryTrace::iterator &it ) const override;
virtual bool isNull() const override;

virtual iterator *clone() const override;

virtual TRecordType    getRecordType() const override;
virtual TRecordTime    getTime() const override;
virtual TThreadOrder   getThread() const override;
virtual TCPUOrder      getCPU() const override;
virtual TObjectOrder   getOrder() const override;
virtual TEventType     getEventType() const override;
virtual TSemanticValue getEventValue() const override;
virtual TEventValue    getEventValueAsIs() const override;
virtual TState         getState() const override;
virtual TRecordTime    getStateEndTime() const override;
virtual TCommID        getCommIndex() const override;

virtual void setTime( const TRecordTime time ) override;
virtual void setRecordType( const TRecordType whichType ) override;
virtual void setStateEndTime( const TRecordTime whichEndTime ) override;

private:
TThreadRecordContainer::iterator it;
VectorBlocks *myBlocks;
TThreadOrder myThread;
};

class CPUIterator : public MemoryTrace::iterator
{
public:
CPUIterator( TCPURecordContainer::iterator whichRecord, const Trace *whichTrace, VectorBlocks *whichBlocks, TCPUOrder whichCPU );
virtual void operator++() override;
virtual void operator--() override;
virtual MemoryTrace::iterator& operator=( const MemoryTrace::iterator& copy ) override;

virtual bool operator==( const MemoryTrace::iterator &it ) const override;
virtual bool operator!=( const MemoryTrace::iterator &it ) const override;
virtual bool isNull() const override;

virtual CPUIterator *clone() const override;

virtual TRecordType    getRecordType() const override;
virtual TRecordTime    getTime() const override;
virtual TThreadOrder   getThread() const override;
virtual TCPUOrder      getCPU() const override;
virtual TObjectOrder   getOrder() const override;
virtual TEventType     getEventType() const override;
virtual TSemanticValue getEventValue() const override;
virtual TEventValue    getEventValueAsIs() const override;
virtual TState         getState() const override;
virtual TRecordTime    getStateEndTime() const override;
virtual TCommID        getCommIndex() const override;

virtual void setTime( const TRecordTime time ) override;
virtual void setRecordType( const TRecordType whichType ) override;
virtual void setStateEndTime( const TRecordTime whichEndTime ) override;

private:
TCPURecordContainer::iterator it;
VectorBlocks *myBlocks;
TCPUOrder myCPU;
};

virtual void insert( MemoryBlocks *blocks ) override;
virtual TTime finish( TTime headerTime, Trace *whichTrace ) override;

virtual MemoryTrace::iterator* empty() const override;
virtual MemoryTrace::iterator* begin() const override;
virtual MemoryTrace::iterator* end() const override;

virtual MemoryTrace::iterator* threadBegin( TThreadOrder whichThread ) const override;
virtual MemoryTrace::iterator* threadEnd( TThreadOrder whichThread ) const override;
virtual MemoryTrace::iterator* CPUBegin( TCPUOrder whichCPU ) const override;
virtual MemoryTrace::iterator* CPUEnd( TCPUOrder whichCPU ) const override;

virtual void getRecordByTimeThread( std::vector<MemoryTrace::iterator *>& listIter,
TRecordTime whichTime ) const override;
virtual void getRecordByTimeCPU( std::vector<MemoryTrace::iterator *>& listIter,
TRecordTime whichTime ) const override;

private:
VectorBlocks *myBlocks;
Trace *myTrace;
};
