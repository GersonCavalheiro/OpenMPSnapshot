#ifndef _DEM_RECORDS_REPOSITORYSTATE_H
#define _DEM_RECORDS_REPOSITORYSTATE_H

#include "tarch/compiler/CompilerSpecificSettings.h"
#include "peano/utils/PeanoOptimisations.h"
#ifdef Parallel
#include "tarch/parallel/Node.h"
#endif
#ifdef Parallel
#include <mpi.h>
#endif
#include "tarch/logging/Log.h"
#include "tarch/la/Vector.h"
#include <bitset>
#include <complex>
#include <string>
#include <iostream>

namespace dem {
namespace records {
class RepositoryState;
class RepositoryStatePacked;
}
}


class dem::records::RepositoryState { 

public:

typedef dem::records::RepositoryStatePacked Packed;

enum Action {
WriteCheckpoint = 0, ReadCheckpoint = 1, Terminate = 2, RunOnAllNodes = 3, UseAdapterCreateGrid = 4, UseAdapterCreateGridAndPlot = 5, UseAdapterTimeStep = 6, UseAdapterTimeStepAndPlot = 7, UseAdapterCollision = 8, UseAdapterMoveParticles = 9, UseAdapterAdopt = 10, UseAdapterAdoptReluctantly = 11, UseAdapterFlopAdopt = 12, UseAdapterPlotData = 13, UseAdapterTimeStepOnDynamicGrid = 14, UseAdapterTimeStepAndPlotOnDynamicGrid = 15, UseAdapterTimeStepOnReluctantDynamicGrid = 16, UseAdapterTimeStepAndPlotOnReluctantDynamicGrid = 17, UseAdapterTimeStepOnFlopDynamicGrid = 18, UseAdapterTimeStepAndPlotOnFlopDynamicGrid = 19, NumberOfAdapters = 20
};

struct PersistentRecords {
Action _action;
int _numberOfIterations;
bool _exchangeBoundaryVertices;

PersistentRecords();


PersistentRecords(const Action& action, const int& numberOfIterations, const bool& exchangeBoundaryVertices);


inline Action getAction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _action;
}



inline void setAction(const Action& action) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_action = action;
}



inline int getNumberOfIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfIterations;
}



inline void setNumberOfIterations(const int& numberOfIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfIterations = numberOfIterations;
}



inline bool getExchangeBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _exchangeBoundaryVertices;
}



inline void setExchangeBoundaryVertices(const bool& exchangeBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_exchangeBoundaryVertices = exchangeBoundaryVertices;
}



};
private: 
PersistentRecords _persistentRecords;

public:

RepositoryState();


RepositoryState(const PersistentRecords& persistentRecords);


RepositoryState(const Action& action, const int& numberOfIterations, const bool& exchangeBoundaryVertices);


virtual ~RepositoryState();


inline Action getAction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._action;
}



inline void setAction(const Action& action) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._action = action;
}



inline int getNumberOfIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfIterations;
}



inline void setNumberOfIterations(const int& numberOfIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfIterations = numberOfIterations;
}



inline bool getExchangeBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._exchangeBoundaryVertices;
}



inline void setExchangeBoundaryVertices(const bool& exchangeBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._exchangeBoundaryVertices = exchangeBoundaryVertices;
}



static std::string toString(const Action& param);


static std::string getActionMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

RepositoryStatePacked convert() const;


#ifdef Parallel
protected:
static tarch::logging::Log _log;

int _senderDestinationRank;

public:


static MPI_Datatype Datatype;
static MPI_Datatype FullDatatype;


static void initDatatype();

static void shutdownDatatype();

enum class ExchangeMode { Blocking, NonblockingWithPollingLoopOverTests, LoopOverProbeWithBlockingReceive };

void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);

int getSenderRank() const;
#endif

};

#ifndef DaStGenPackedPadding
#define DaStGenPackedPadding 1      
#endif


#ifdef PackedRecords
#pragma pack (push, DaStGenPackedPadding)
#endif


class dem::records::RepositoryStatePacked { 

public:

typedef dem::records::RepositoryState::Action Action;

struct PersistentRecords {
Action _action;
int _numberOfIterations;
bool _exchangeBoundaryVertices;

PersistentRecords();


PersistentRecords(const Action& action, const int& numberOfIterations, const bool& exchangeBoundaryVertices);


inline Action getAction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _action;
}



inline void setAction(const Action& action) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_action = action;
}



inline int getNumberOfIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfIterations;
}



inline void setNumberOfIterations(const int& numberOfIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfIterations = numberOfIterations;
}



inline bool getExchangeBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _exchangeBoundaryVertices;
}



inline void setExchangeBoundaryVertices(const bool& exchangeBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_exchangeBoundaryVertices = exchangeBoundaryVertices;
}



};
private: 
PersistentRecords _persistentRecords;

public:

RepositoryStatePacked();


RepositoryStatePacked(const PersistentRecords& persistentRecords);


RepositoryStatePacked(const Action& action, const int& numberOfIterations, const bool& exchangeBoundaryVertices);


virtual ~RepositoryStatePacked();


inline Action getAction() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._action;
}



inline void setAction(const Action& action) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._action = action;
}



inline int getNumberOfIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfIterations;
}



inline void setNumberOfIterations(const int& numberOfIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfIterations = numberOfIterations;
}



inline bool getExchangeBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._exchangeBoundaryVertices;
}



inline void setExchangeBoundaryVertices(const bool& exchangeBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._exchangeBoundaryVertices = exchangeBoundaryVertices;
}



static std::string toString(const Action& param);


static std::string getActionMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

RepositoryState convert() const;


#ifdef Parallel
protected:
static tarch::logging::Log _log;

int _senderDestinationRank;

public:


static MPI_Datatype Datatype;
static MPI_Datatype FullDatatype;


static void initDatatype();

static void shutdownDatatype();

enum class ExchangeMode { Blocking, NonblockingWithPollingLoopOverTests, LoopOverProbeWithBlockingReceive };

void send(int destination, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

void receive(int source, int tag, bool exchangeOnlyAttributesMarkedWithParallelise, ExchangeMode mode );

static bool isMessageInQueue(int tag, bool exchangeOnlyAttributesMarkedWithParallelise);

int getSenderRank() const;
#endif

};

#ifdef PackedRecords
#pragma pack (pop)
#endif


#endif

