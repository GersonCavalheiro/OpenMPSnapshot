

#pragma once

#include <array>

#include "memoryblocks.h"
#include "vectortrace.h"
#include "vectortracedefines.h"

#include "utils/traceparser/processmodel.h"
#include "utils/traceparser/resourcemodel.h"
#include "utils/traceparser/tracetypes.h"


constexpr size_t BEGIN_EMPTY_RECORD = 0;
constexpr size_t END_EMPTY_RECORD = 1;

class VectorBlocks : public MemoryBlocks
{
public:
VectorBlocks( const ResourceModel<>& resource,
const ProcessModel<>& process,
TRecordTime endTime,
ProgressController *whichProgress );

virtual TData *getLastRecord( PRV_UINT16 position ) const override;
virtual void newRecord() override;
virtual void newRecord( TThreadOrder whichThread ) override;
virtual void setRecordType( TRecordType whichType ) override;
virtual void setTime( TRecordTime whichTime ) override;
virtual void setThread( TThreadOrder whichThread ) override;
virtual void setThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) override;
virtual void setCPU( TCPUOrder whichCPU ) override;
virtual void setEventType( TEventType whichType ) override;
virtual void setEventValue( TEventValue whichValue ) override;
virtual void setState( TState whichState ) override;
virtual void setStateEndTime( TRecordTime whichTime ) override;
virtual void setCommIndex( TCommID whichID ) override;

virtual void newComm( bool createRecords = true ) override;
virtual void newComm( TThreadOrder whichSenderThread, TThreadOrder whichReceiverThread, bool createRecords = true ) override;
virtual void setSenderThread( TThreadOrder whichThread ) override;
virtual void setSenderThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) override;
virtual void setSenderCPU( TCPUOrder whichCPU ) override;
virtual void setReceiverThread( TThreadOrder whichThread ) override;
virtual void setReceiverThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) override;
virtual void setReceiverCPU( TCPUOrder whichCPU ) override;
virtual void setCommTag( TCommTag whichTag ) override;
virtual void setCommSize( TCommSize whichSize ) override;
virtual void setLogicalSend( TRecordTime whichTime ) override;
virtual void setLogicalReceive( TRecordTime whichTime ) override;
virtual void setPhysicalSend( TRecordTime whichTime ) override;
virtual void setPhysicalReceive( TRecordTime whichTime ) override;

virtual TCommID getTotalComms() const override;
virtual TThreadOrder getSenderThread( TCommID whichComm ) const override;
virtual TCPUOrder getSenderCPU( TCommID whichComm ) const override;
virtual TThreadOrder getReceiverThread( TCommID whichComm ) const override;
virtual TCPUOrder getReceiverCPU( TCommID whichComm ) const override;
virtual TCommTag getCommTag( TCommID whichComm ) const override;
virtual TCommSize getCommSize( TCommID whichComm ) const override;
virtual TRecordTime getLogicalSend( TCommID whichComm ) const override;
virtual TRecordTime getLogicalReceive( TCommID whichComm ) const override;
virtual TRecordTime getPhysicalSend( TCommID whichComm ) const override;
virtual TRecordTime getPhysicalReceive( TCommID whichComm ) const override;

virtual TRecordTime getLastRecordTime() const override;

virtual void setFileLoaded( TRecordTime traceEndTime ) override;

private:
std::vector< TThreadRecordContainer > threadRecords;
std::vector< TCPURecordContainer > cpuRecords;

std::vector< Plain::TRecord > cpuBeginEmptyRecords;
std::vector< Plain::TRecord > cpuEndEmptyRecords;

typedef enum
{
logicalSend = 0,
logicalReceive,
physicalSend,
physicalReceive,
remoteLogicalSend,
remoteLogicalReceive,
remotePhysicalSend,
remotePhysicalReceive,
commTypeSize
} TCommType;
static const TRecordType commTypes[ commTypeSize ];
std::vector<Plain::TCommInfo> communications;
std::array<Plain::TRecord*, commTypeSize> commRecords;

const ResourceModel<>& resourceModel;
const ProcessModel<>& processModel;

TThreadOrder insertedOnThread;
TRecordTime lastRecordTime = 0;

ProgressController *progress;

friend class VectorTrace;
friend class VectorTrace::iterator;

};