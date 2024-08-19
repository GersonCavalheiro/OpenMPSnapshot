


#pragma once


#include <map>
#include <vector>
#include <string>
#include "paraverkerneltypes.h"
#include "traceeditstates.h"

using std::map;
using std::vector;

class KernelConnection;
class TraceEditAction;

enum class TSequenceStates
{
testState = 0,
traceOptionsState,
sourceTimelineState,
csvFileNameState,
csvOutputState,
outputDirSuffixState,
outputTraceFileNameState,
maxTraceTimeState,
shiftTimesState,
eofParsedState,
shiftLevelState,
onEventCutterState,
pcfMergerReferenceState,
eventTranslationTableState,
copyAdditionalFilesState,
onlyFilterState,
numStates
};

enum class TSequenceActions
{
testAction = 0,
traceCutterAction,
traceFilterAction,
csvOutputAction,
traceParserAction,
recordTimeShifterAction,
traceWriterAction,
eventDrivenCutterAction,
traceSortAction,
numActions
};

class TraceEditSequence
{
public:
static std::string dirNameClustering;
static std::string dirNameFolding;
static std::string dirNameDimemas;
static std::string dirNameSpectral;
static std::string dirNameProfet;
static std::string dirNameUserCommand;

static TraceEditSequence *create( const KernelConnection *whichKernel );

TraceEditSequence() {}
TraceEditSequence( const KernelConnection *whichKernel ) {}
virtual ~TraceEditSequence() {}

virtual const KernelConnection *getKernelConnection() const = 0;

virtual TraceEditState *createState( TSequenceStates whichState ) = 0;

virtual bool addState( TSequenceStates whichState ) = 0;
virtual bool addState( TSequenceStates whichState, TraceEditState *newState ) = 0;
virtual TraceEditState *getState( TSequenceStates whichState ) = 0;
virtual bool pushbackAction( TSequenceActions whichAction ) = 0;
virtual bool pushbackAction( TraceEditAction *newAction ) = 0;

virtual bool execute( vector<std::string> traces ) = 0;

virtual bool executeNextAction( std::string whichTrace ) = 0;

virtual bool isEndOfSequence() const = 0;

virtual TraceEditSequence *getConcrete()
{
return nullptr;
}

protected:

private:

};


class TraceEditSequenceProxy:public TraceEditSequence
{
public:
TraceEditSequenceProxy() {}
TraceEditSequenceProxy( const KernelConnection *whichKernel );
virtual ~TraceEditSequenceProxy();

const KernelConnection *getKernelConnection() const override;

TraceEditState *createState( TSequenceStates whichState ) override;

bool addState( TSequenceStates whichState ) override;
bool addState( TSequenceStates whichState, TraceEditState *newState ) override;
TraceEditState *getState( TSequenceStates whichState ) override;
bool pushbackAction( TSequenceActions whichAction ) override;
bool pushbackAction( TraceEditAction *newAction ) override;

bool execute( vector<std::string> traces ) override;

bool executeNextAction( std::string whichTrace ) override;

bool isEndOfSequence() const override;

TraceEditSequence *getConcrete() override;

protected:

private:
TraceEditSequence *mySequence;

};


