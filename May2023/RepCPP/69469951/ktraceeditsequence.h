


#pragma once


#include "traceeditsequence.h"
#include "ktrace.h"

class KTrace;

class KTraceEditSequence:public TraceEditSequence
{
public:
KTraceEditSequence() {}
KTraceEditSequence( const KernelConnection *whichKernel );
virtual ~KTraceEditSequence();

const KernelConnection *getKernelConnection() const override;

TraceEditState *createState( TSequenceStates whichState ) override;
void setCurrentTrace( KTrace *whichTrace );
KTrace *getCurrentTrace();

bool addState( TSequenceStates whichState ) override;
bool addState( TSequenceStates whichState, TraceEditState *newState ) override;
TraceEditState *getState( TSequenceStates whichState ) override;
bool pushbackAction( TSequenceActions whichAction ) override;
bool pushbackAction( TraceEditAction *newAction ) override;

bool execute( vector<std::string> traces ) override;

bool executeNextAction( std::string whichTrace ) override;
bool executeNextAction( MemoryTrace::iterator *whichRecord );

bool isEndOfSequence() const override;

protected:

private:
map<TSequenceStates, TraceEditState *> activeStates;
vector<TraceEditAction *> sequenceActions;
KTrace *currentTrace;
std::string currentTraceName;
map< std::string, bool > sequenceExecError;

PRV_UINT16 currentAction;

const KernelConnection *myKernel;

};



