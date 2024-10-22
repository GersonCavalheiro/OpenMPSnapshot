

#pragma once


#include <set>
#include "boost/date_time/posix_time/posix_time.hpp"

#include "paraverkerneltypes.h"
#include "semanticcolor.h"
#include "eventlabels.h"
#include "statelabels.h"
#include "utils/traceparser/rowfileparser.h"

using boost::posix_time::ptime;

class KernelConnection;
class ProgressController;

class Trace
{
public:
static Trace *create( KernelConnection *whichKernel, const std::string& whichFile,
bool noLoad, ProgressController *progress );
static bool isOTF2TraceFile( const std::string& filename );


Trace() {}
Trace( KernelConnection *whichKernel );
virtual ~Trace() {}

TObjectOrder getLevelObjects( TTraceLevel onLevel ) const;

virtual std::string getFileName() const = 0;
virtual std::string getTraceName() const = 0;
virtual TTraceSize getTraceSize() const = 0;

virtual void dumpFileHeader( std::fstream& file, bool newFormat = false ) const = 0;
virtual void dumpFile( const std::string& whichFile ) const = 0;

virtual TApplOrder totalApplications() const = 0;
virtual TTaskOrder totalTasks() const = 0;
virtual TTaskOrder getGlobalTask( const TApplOrder& inAppl,
const TTaskOrder& inTask ) const = 0;
virtual void getTaskLocation( TTaskOrder globalTask,
TApplOrder& inAppl,
TTaskOrder& inTask ) const = 0;
virtual TTaskOrder getFirstTask( TApplOrder inAppl ) const = 0;
virtual TTaskOrder getLastTask( TApplOrder inAppl ) const = 0;

virtual TThreadOrder totalThreads() const = 0;
virtual TThreadOrder getGlobalThread( const TApplOrder& inAppl,
const TTaskOrder& inTask,
const TThreadOrder& inThread ) const = 0;
virtual void getThreadLocation( TThreadOrder globalThread,
TApplOrder& inAppl,
TTaskOrder& inTask,
TThreadOrder& inThread ) const = 0;
virtual TThreadOrder getFirstThread( TApplOrder inAppl, TTaskOrder inTask ) const = 0;
virtual TThreadOrder getLastThread( TApplOrder inAppl, TTaskOrder inTask ) const = 0;
virtual void getThreadsPerNode( TNodeOrder inNode, std::vector<TThreadOrder>& onVector ) const = 0;

virtual TNodeOrder getNodeFromThread( TThreadOrder &whichThread ) const = 0;

virtual bool existResourceInfo() const = 0;
virtual TNodeOrder totalNodes() const = 0;
virtual TCPUOrder totalCPUs() const = 0;
virtual TCPUOrder getGlobalCPU( const TNodeOrder& inNode,
const TCPUOrder& inCPU ) const = 0;
virtual void getCPULocation( TCPUOrder globalCPU,
TNodeOrder& inNode,
TCPUOrder& inCPU ) const = 0;
virtual TCPUOrder getFirstCPU( TNodeOrder inNode ) const = 0;
virtual TCPUOrder getLastCPU( TNodeOrder inNode ) const = 0;

virtual TObjectOrder getFirst( TObjectOrder globalOrder,
TTraceLevel fromLevel,
TTraceLevel toLevel ) const = 0;
virtual TObjectOrder getLast( TObjectOrder globalOrder,
TTraceLevel fromLevel,
TTraceLevel toLevel ) const = 0;

virtual bool isSameObjectStruct( Trace *compareTo, bool compareProcessModel ) const = 0;
virtual bool isSubsetObjectStruct( Trace *compareTo, bool compareProcessModel ) const = 0;

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

virtual TTime getEndTime() const = 0;
virtual void setEndTime( TTime whichTraceEndTime ) = 0;
virtual TTimeUnit getTimeUnit() const = 0;
virtual ptime getTraceTime() const = 0;

virtual TRecordTime customUnitsToTraceUnits( TRecordTime whichTime, TTimeUnit whichUnits ) const = 0;
virtual TRecordTime traceUnitsToCustomUnits( TRecordTime whichTime, TTimeUnit whichUnits ) const = 0;

virtual bool eventLoaded( TEventType whichType ) const = 0;
virtual bool anyEventLoaded( TEventType firstType, TEventType lastType ) const = 0;
virtual const std::set<TState>& getLoadedStates() const = 0;
virtual const std::set<TEventType>& getLoadedEvents() const = 0;

virtual bool findLastEventValue( TThreadOrder whichThread,
TRecordTime whichTime,
const std::vector<TEventType>& whichEvent,
TEventType& returnType,
TEventValue& returnValue ) const = 0;

virtual bool findNextEvent( TThreadOrder whichThread,
TRecordTime whichTime,
TEventType whichEvent,
TRecordTime& foundTime ) const = 0;

virtual bool getFillStateGaps() const = 0;

virtual void setFillStateGaps( bool fill ) = 0;

virtual PRV_UINT64 getCutterOffset() = 0;
virtual PRV_UINT64 getCutterLastOffset() = 0;
virtual PRV_UINT64 getCutterLastBeginTime() = 0;
virtual PRV_UINT64 getCutterLastEndTime() = 0;
virtual PRV_UINT64 getCutterBeginTime() = 0;
virtual PRV_UINT64 getCutterEndTime() = 0;

virtual void setLogicalSend( TCommID whichComm, TRecordTime whichTime ) = 0;
virtual void setLogicalReceive( TCommID whichComm, TRecordTime whichTime ) = 0;
virtual void setPhysicalSend( TCommID whichComm, TRecordTime whichTime ) = 0;
virtual void setPhysicalReceive( TCommID whichComm, TRecordTime whichTime ) = 0;

virtual void   setEventTypePrecision( TEventType whichType, double whichPrecision ) = 0;
virtual double getEventTypePrecision( TEventType whichType ) const = 0;

virtual bool getUnload() const
{
return false;
}
virtual void setUnload( bool newValue ) {}
virtual Trace *getConcrete() const
{
return nullptr;
}
virtual std::string getFileNameNumbered() const
{
return "";
}
virtual std::string getTraceNameNumbered() const
{
return "";
}
virtual void setInstanceNumber( PRV_UINT32 whichInstanceNumber ) {}
virtual const CodeColor& getCodeColor() const
{
CodeColor *tmp = nullptr;
return *tmp;
}
virtual const EventLabels& getEventLabels() const
{
EventLabels *tmp = nullptr;
return *tmp;
}
virtual const StateLabels& getStateLabels() const
{
StateLabels *tmp = nullptr;
return *tmp;
}
virtual std::string getRowLabel( TTraceLevel whichLevel, TObjectOrder whichRow ) const
{
return "";
}

virtual size_t getMaxLengthRow( TTraceLevel whichLevel = TTraceLevel::NONE ) const
{
return 0;
}

virtual std::string getDefaultSemanticFunc( TWindowLevel whichLevel ) const
{
return "";
}

virtual void setShowProgressBar( bool whichShow )
{}

virtual bool getShowProgressBar() const
{
return true;
}



protected:
KernelConnection *myKernel;

};

class TraceProxy: public Trace
{
public:
virtual ~TraceProxy();

virtual std::string getFileName() const override;
virtual std::string getTraceName() const override;
virtual std::string getFileNameNumbered() const override;
virtual std::string getTraceNameNumbered() const override;
virtual void setInstanceNumber( PRV_UINT32 whichInstanceNumber ) override;

virtual void dumpFileHeader( std::fstream& file, bool newFormat = false ) const override;
virtual void dumpFile( const std::string& whichFile ) const override;

virtual TTraceSize getTraceSize() const override;

virtual TApplOrder totalApplications() const override;
virtual TTaskOrder totalTasks() const override;
virtual TTaskOrder getGlobalTask( const TApplOrder& inAppl,
const TTaskOrder& inTask ) const override;
virtual void getTaskLocation( TTaskOrder globalTask,
TApplOrder& inAppl,
TTaskOrder& inTask ) const override;
virtual TTaskOrder getFirstTask( TApplOrder inAppl ) const override;
virtual TTaskOrder getLastTask( TApplOrder inAppl ) const override;

virtual TThreadOrder totalThreads() const override;
virtual TThreadOrder getGlobalThread( const TApplOrder& inAppl,
const TTaskOrder& inTask,
const TThreadOrder& inThread ) const override;
virtual void getThreadLocation( TThreadOrder globalThread,
TApplOrder& inAppl,
TTaskOrder& inTask,
TThreadOrder& inThread ) const override;
virtual TThreadOrder getFirstThread( TApplOrder inAppl, TTaskOrder inTask ) const override;
virtual TThreadOrder getLastThread( TApplOrder inAppl, TTaskOrder inTask ) const override;
virtual void getThreadsPerNode( TNodeOrder inNode, std::vector<TThreadOrder> &onVector ) const override;

virtual TNodeOrder getNodeFromThread( TThreadOrder &whichThread ) const override;

virtual bool existResourceInfo() const override;
virtual TNodeOrder totalNodes() const override;
virtual TCPUOrder totalCPUs() const override;
virtual TCPUOrder getGlobalCPU( const TNodeOrder& inNode,
const TCPUOrder& inCPU ) const override;
virtual void getCPULocation( TCPUOrder globalCPU,
TNodeOrder& inNode,
TCPUOrder& inCPU ) const override;
virtual TCPUOrder getFirstCPU( TNodeOrder inNode ) const override;
virtual TCPUOrder getLastCPU( TNodeOrder inNode ) const override;

virtual TObjectOrder getFirst( TObjectOrder globalOrder,
TTraceLevel fromLevel,
TTraceLevel toLevel ) const override;
virtual TObjectOrder getLast( TObjectOrder globalOrder,
TTraceLevel fromLevel,
TTraceLevel toLevel ) const override;

virtual bool isSameObjectStruct( Trace *compareTo, bool compareProcessModel ) const override;
virtual bool isSubsetObjectStruct( Trace *compareTo, bool compareProcessModel ) const override;

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

virtual TTime getEndTime() const override;
virtual void setEndTime( TTime whichTraceEndTime ) override;
virtual TTimeUnit getTimeUnit() const override;
virtual ptime getTraceTime() const override;

virtual TRecordTime customUnitsToTraceUnits( TRecordTime whichTime, TTimeUnit whichUnits ) const override;
virtual TRecordTime traceUnitsToCustomUnits( TRecordTime whichTime, TTimeUnit whichUnits ) const override;

virtual bool getUnload() const override;
virtual void setUnload( bool newValue ) override;
virtual Trace *getConcrete() const override;
virtual const CodeColor& getCodeColor() const override;
virtual const EventLabels& getEventLabels() const override;
virtual const StateLabels& getStateLabels() const override;
virtual std::string getRowLabel( TTraceLevel whichLevel, TObjectOrder whichRow ) const override;
virtual size_t getMaxLengthRow( TTraceLevel whichLevel ) const override;

virtual void setShowProgressBar( bool whichShow ) override;
virtual bool getShowProgressBar() const override;

virtual bool eventLoaded( TEventType whichType ) const override;
virtual bool anyEventLoaded( TEventType firstType, TEventType lastType ) const override;
virtual const std::set<TState>& getLoadedStates() const override;
virtual const std::set<TEventType>& getLoadedEvents() const override;

virtual bool findLastEventValue( TThreadOrder whichThread,
TRecordTime whichTime,
const std::vector<TEventType>& whichEvent,
TEventType& returnType,
TEventValue& returnValue ) const override;

virtual bool findNextEvent( TThreadOrder whichThread,
TRecordTime whichTime,
TEventType whichEvent,
TRecordTime& foundTime ) const override;

virtual bool getFillStateGaps() const override;
virtual void setFillStateGaps( bool fill ) override;

virtual PRV_UINT64 getCutterOffset() override;
virtual PRV_UINT64 getCutterLastOffset() override;
virtual PRV_UINT64 getCutterLastBeginTime() override;
virtual PRV_UINT64 getCutterLastEndTime() override;
virtual PRV_UINT64 getCutterBeginTime() override;
virtual PRV_UINT64 getCutterEndTime() override;

virtual void setLogicalSend( TCommID whichComm, TRecordTime whichTime ) override;
virtual void setLogicalReceive( TCommID whichComm, TRecordTime whichTime ) override;
virtual void setPhysicalSend( TCommID whichComm, TRecordTime whichTime ) override;
virtual void setPhysicalReceive( TCommID whichComm, TRecordTime whichTime ) override;

virtual void   setEventTypePrecision( TEventType whichType, double whichPrecision ) override;
virtual double getEventTypePrecision( TEventType whichType ) const override;

private:
Trace *myTrace;

bool unload;
PRV_UINT32 instanceNumber;

CodeColor myCodeColor;

EventLabels myEventLabels;
StateLabels myStateLabels;
RowFileParser<> myRowLabels;

bool showProgressBar;


TraceProxy( KernelConnection *whichKernel, const std::string& whichFile,
bool noLoad, ProgressController *progress );

void parsePCF( const std::string& whichFile );
void parseROW( const std::string& whichFile );

#ifdef FIXED_LABELS
void setFixedLabels();
#endif

friend Trace *Trace::create( KernelConnection *, const std::string&, bool noLoad, ProgressController * );
};


