


#pragma once


#include <fstream>

#include "traceeditactions.h"
#include "memorytrace.h"
#include "tracebodyiofactory.h"
#include "utils/traceparser/tracebodyio_v1.h"

class TraceToRecordAction: public TraceEditAction
{
public:
TraceToRecordAction( TraceEditSequence *whichSequence ) : TraceEditAction( whichSequence )
{}
~TraceToRecordAction()
{}

virtual TTraceEditActionType getType() const override
{
return TTraceEditActionType::TraceToRecord;
}

virtual bool execute( std::string whichTrace ) = 0;

protected:

private:

};


class RecordToTraceAction: public TraceEditAction
{
public:
RecordToTraceAction( TraceEditSequence *whichSequence ) : TraceEditAction( whichSequence )
{}
~RecordToTraceAction()
{}

virtual TTraceEditActionType getType() const override
{
return TTraceEditActionType::RecordToTrace;
}

virtual bool execute( MemoryTrace::iterator *whichRecord ) = 0;

protected:

private:

};

class RecordToRecordAction: public TraceEditAction
{
public:
RecordToRecordAction( TraceEditSequence *whichSequence ) : TraceEditAction( whichSequence )
{}
~RecordToRecordAction()
{}

virtual TTraceEditActionType getType() const override
{
return TTraceEditActionType::RecordToRecord;
}

virtual bool execute( MemoryTrace::iterator *whichRecord ) = 0;

protected:

private:

};



class TestAction: public TraceToTraceAction
{
public:
TestAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~TestAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class TraceCutterAction: public TraceToTraceAction
{
public:
TraceCutterAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~TraceCutterAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class TraceFilterAction: public TraceToTraceAction
{
public:
TraceFilterAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~TraceFilterAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};



class CSVOutputAction: public TraceToTraceAction
{
public:
CSVOutputAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~CSVOutputAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};






class TraceParserAction: public TraceToRecordAction
{
public:
TraceParserAction( TraceEditSequence *whichSequence ) : TraceToRecordAction( whichSequence )
{}
~TraceParserAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:
};



class RecordTimeShifterAction: public RecordToRecordAction
{
public:
RecordTimeShifterAction( TraceEditSequence *whichSequence ) :
RecordToRecordAction( whichSequence ), availableShiftTime( false ), checkedAvailableShiftTime( false ),
objects( TObjectOrder(0) ), count( 0 )
{}
~RecordTimeShifterAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( MemoryTrace::iterator *whichRecord ) override;

protected:

private:
bool availableShiftTime;
bool checkedAvailableShiftTime;
TObjectOrder objects;
int count;

};



class TraceWriterAction: public RecordToTraceAction
{
public:
TraceWriterAction( TraceEditSequence *whichSequence ) : RecordToTraceAction( whichSequence )
{}
~TraceWriterAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( MemoryTrace::iterator *whichRecord ) override;

protected:

private:
std::fstream outputTrace;
TraceBodyIO_v1< PARAM_TRACEBODY_CLASS > body;
};



class EventDrivenCutterAction: public RecordToTraceAction
{
public:
EventDrivenCutterAction( TraceEditSequence *whichSequence ) : RecordToTraceAction( whichSequence )
{}
~EventDrivenCutterAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( MemoryTrace::iterator *whichRecord ) override;

protected:

private:
vector<std::fstream *> outputTraces;
vector<PRV_UINT32> currentThreadFile;
map<PRV_INT32, TObjectOrder> countThreadsPerFile;
TraceBodyIO_v1< PARAM_TRACEBODY_CLASS > body;
};


class TraceSortAction: public TraceToTraceAction
{
public:
TraceSortAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~TraceSortAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};


class PCFEventMergerAction: public TraceToTraceAction
{
public:
PCFEventMergerAction( TraceEditSequence *whichSequence ) : TraceToTraceAction( whichSequence )
{}
~PCFEventMergerAction()
{}

virtual vector<TSequenceStates> getStateDependencies() const override;

virtual bool execute( std::string whichTrace ) override;

protected:

private:

};




