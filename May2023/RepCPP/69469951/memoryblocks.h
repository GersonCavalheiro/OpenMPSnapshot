


#pragma once


#include "paraverkerneltypes.h"
#include "memorytrace.h"

class MemoryBlocks
{
public:
MemoryBlocks()
{
countInserted = 0;
}

virtual ~MemoryBlocks()
{}

virtual TData *getLastRecord( PRV_UINT16 position ) const = 0;
virtual void newRecord() = 0;
virtual void newRecord( TThreadOrder whichThread ) = 0;
virtual void setRecordType( TRecordType whichType ) = 0;
virtual void setTime( TRecordTime whichTime ) = 0;
virtual void setThread( TThreadOrder whichThread ) = 0;
virtual void setThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) = 0;
virtual void setCPU( TCPUOrder whichCPU ) = 0;
virtual void setEventType( TEventType whichType ) = 0;
virtual void setEventValue( TEventValue whichValue ) = 0;
virtual void setState( TState whichState ) = 0;
virtual void setStateEndTime( TRecordTime whichTime ) = 0;
virtual void setCommIndex( TCommID whichID ) = 0;

virtual void newComm( bool createRecords = true ) = 0;
virtual void newComm( TThreadOrder whichSenderThread, TThreadOrder whichReceiverThread, bool createRecords = true ) = 0;
virtual void setSenderThread( TThreadOrder whichThread ) = 0;
virtual void setSenderThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) = 0;
virtual void setSenderCPU( TCPUOrder whichCPU ) = 0;
virtual void setReceiverThread( TThreadOrder whichThread ) = 0;
virtual void setReceiverThread( TApplOrder whichAppl,
TTaskOrder whichTask,
TThreadOrder whichThread ) = 0;
virtual void setReceiverCPU( TCPUOrder whichCPU ) = 0;
virtual void setCommTag( TCommTag whichTag ) = 0;
virtual void setCommSize( TCommSize whichSize ) = 0;
virtual void setLogicalSend( TRecordTime whichTime ) = 0;
virtual void setLogicalReceive( TRecordTime whichTime ) = 0;
virtual void setPhysicalSend( TRecordTime whichTime ) = 0;
virtual void setPhysicalReceive( TRecordTime whichTime ) = 0;
virtual void setLogicalSend( TCommID whichComm, TRecordTime whichTime )
{}
virtual void setLogicalReceive( TCommID whichComm, TRecordTime whichTime )
{}
virtual void setPhysicalSend( TCommID whichComm, TRecordTime whichTime )
{}
virtual void setPhysicalReceive( TCommID whichComm, TRecordTime whichTime )
{}

virtual TCommID getTotalComms() const = 0;
virtual TThreadOrder getSenderThread( TCommID whichComm ) const = 0;
virtual TCPUOrder getSenderCPU( TCommID whichComm ) const = 0;
virtual TThreadOrder getReceiverThread( TCommID whichComm ) const = 0;
virtual TCPUOrder getReceiverCPU( TCommID whichComm ) const = 0;
virtual TCommTag getCommTag( TCommID whichComm ) const = 0;
virtual TCommSize getCommSize( TCommID whichComm ) const = 0;
virtual TRecordTime getLogicalSend( TCommID whichComm ) const = 0;
virtual TRecordTime getLogicalReceive( TCommID whichComm ) const = 0;
virtual TRecordTime getPhysicalSend( TCommID whichComm ) const = 0;
virtual TRecordTime getPhysicalReceive( TCommID whichComm ) const = 0;

virtual TRecordTime getLastRecordTime() const = 0;

virtual PRV_UINT32 getCountInserted() const
{
return countInserted;
}

virtual void resetCountInserted()
{
countInserted = 0;
}

virtual void setFileLoaded( TRecordTime traceEndTime )
{}

protected:
PRV_UINT32 countInserted;

private:
};



