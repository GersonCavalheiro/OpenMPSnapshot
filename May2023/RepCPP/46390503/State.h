#ifndef _DEM_RECORDS_STATE_H
#define _DEM_RECORDS_STATE_H

#include "peano/utils/Globals.h"
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
class State;
class StatePacked;
}
}

#if !defined(TrackGridStatistics) && !defined(Parallel)

class dem::records::State { 

public:

typedef dem::records::StatePacked Packed;

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
bool _hasRefined;
bool _hasTriggeredRefinementForNextIteration;
bool _hasErased;
bool _hasTriggeredEraseForNextIteration;
bool _hasChangedVertexOrCellState;
bool _hasModifiedGridInPreviousIteration;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;

PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

State();


State(const PersistentRecords& persistentRecords);


State(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~State();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

StatePacked convert() const;


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


class dem::records::StatePacked { 

public:

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( hasRefined ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_packedRecords0 = static_cast<short int>( hasErased ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

StatePacked();


StatePacked(const PersistentRecords& persistentRecords);


StatePacked(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~StatePacked();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( hasRefined ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_persistentRecords._packedRecords0 = static_cast<short int>( hasErased ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_persistentRecords._packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_persistentRecords._packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

State convert() const;


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


#elif defined(TrackGridStatistics) && defined(Parallel)

class dem::records::State { 

public:

typedef dem::records::StatePacked Packed;

enum BoundaryRefinement {
RefineArtificially = 0, Nop = 1, EraseAggressively = 2
};

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth;
#endif
double _numberOfInnerVertices;
double _numberOfBoundaryVertices;
double _numberOfOuterVertices;
double _numberOfInnerCells;
double _numberOfOuterCells;
double _numberOfInnerLeafVertices;
double _numberOfBoundaryLeafVertices;
double _numberOfOuterLeafVertices;
double _numberOfInnerLeafCells;
double _numberOfOuterLeafCells;
int _maxLevel;
bool _hasRefined;
bool _hasTriggeredRefinementForNextIteration;
bool _hasErased;
bool _hasTriggeredEraseForNextIteration;
bool _hasChangedVertexOrCellState;
bool _hasModifiedGridInPreviousIteration;
bool _isTraversalInverted;
bool _reduceStateAndCell;
bool _couldNotEraseDueToDecompositionFlag;
bool _subWorkerIsInvolvedInJoinOrFork;
BoundaryRefinement _refineArtificiallyOutsideDomain;
int _totalNumberOfBatchIterations;
int _batchIteration;

PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_minMeshWidth = (minMeshWidth);
}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxMeshWidth = (maxMeshWidth);
}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _reduceStateAndCell;
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_reduceStateAndCell = reduceStateAndCell;
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _couldNotEraseDueToDecompositionFlag;
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_couldNotEraseDueToDecompositionFlag = couldNotEraseDueToDecompositionFlag;
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subWorkerIsInvolvedInJoinOrFork;
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subWorkerIsInvolvedInJoinOrFork = subWorkerIsInvolvedInJoinOrFork;
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refineArtificiallyOutsideDomain;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refineArtificiallyOutsideDomain = refineArtificiallyOutsideDomain;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

State();


State(const PersistentRecords& persistentRecords);


State(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~State();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._minMeshWidth = (minMeshWidth);
}



inline double getMinMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._minMeshWidth[elementIndex];

}



inline void setMinMeshWidth(int elementIndex, const double& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._minMeshWidth[elementIndex]= minMeshWidth;

}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxMeshWidth = (maxMeshWidth);
}



inline double getMaxMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._maxMeshWidth[elementIndex];

}



inline void setMaxMeshWidth(int elementIndex, const double& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._maxMeshWidth[elementIndex]= maxMeshWidth;

}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._reduceStateAndCell;
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._reduceStateAndCell = reduceStateAndCell;
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._couldNotEraseDueToDecompositionFlag;
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._couldNotEraseDueToDecompositionFlag = couldNotEraseDueToDecompositionFlag;
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subWorkerIsInvolvedInJoinOrFork;
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subWorkerIsInvolvedInJoinOrFork = subWorkerIsInvolvedInJoinOrFork;
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refineArtificiallyOutsideDomain;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refineArtificiallyOutsideDomain = refineArtificiallyOutsideDomain;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



static std::string toString(const BoundaryRefinement& param);


static std::string getBoundaryRefinementMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

StatePacked convert() const;


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


class dem::records::StatePacked { 

public:

typedef dem::records::State::BoundaryRefinement BoundaryRefinement;

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth;
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth;
double _numberOfInnerVertices;
double _numberOfBoundaryVertices;
double _numberOfOuterVertices;
double _numberOfInnerCells;
double _numberOfOuterCells;
double _numberOfInnerLeafVertices;
double _numberOfBoundaryLeafVertices;
double _numberOfOuterLeafVertices;
double _numberOfInnerLeafCells;
double _numberOfOuterLeafCells;
int _maxLevel;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_minMeshWidth = (minMeshWidth);
}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxMeshWidth = (maxMeshWidth);
}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( hasRefined ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_packedRecords0 = static_cast<short int>( hasErased ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( reduceStateAndCell ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( couldNotEraseDueToDecompositionFlag ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_packedRecords0 = static_cast<short int>( subWorkerIsInvolvedInJoinOrFork ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (9));
assertion(( tmp >= 0 &&  tmp <= 2));
return (BoundaryRefinement) tmp;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refineArtificiallyOutsideDomain >= 0 && refineArtificiallyOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refineArtificiallyOutsideDomain) << (9));
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

StatePacked();


StatePacked(const PersistentRecords& persistentRecords);


StatePacked(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~StatePacked();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._minMeshWidth = (minMeshWidth);
}



inline double getMinMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._minMeshWidth[elementIndex];

}



inline void setMinMeshWidth(int elementIndex, const double& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._minMeshWidth[elementIndex]= minMeshWidth;

}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxMeshWidth = (maxMeshWidth);
}



inline double getMaxMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._maxMeshWidth[elementIndex];

}



inline void setMaxMeshWidth(int elementIndex, const double& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._maxMeshWidth[elementIndex]= maxMeshWidth;

}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( hasRefined ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_persistentRecords._packedRecords0 = static_cast<short int>( hasErased ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_persistentRecords._packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_persistentRecords._packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( reduceStateAndCell ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( couldNotEraseDueToDecompositionFlag ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_persistentRecords._packedRecords0 = static_cast<short int>( subWorkerIsInvolvedInJoinOrFork ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (9));
assertion(( tmp >= 0 &&  tmp <= 2));
return (BoundaryRefinement) tmp;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refineArtificiallyOutsideDomain >= 0 && refineArtificiallyOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refineArtificiallyOutsideDomain) << (9));
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



static std::string toString(const BoundaryRefinement& param);


static std::string getBoundaryRefinementMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

State convert() const;


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



#elif defined(TrackGridStatistics) && !defined(Parallel)

class dem::records::State { 

public:

typedef dem::records::StatePacked Packed;

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth;
#endif
double _numberOfInnerVertices;
double _numberOfBoundaryVertices;
double _numberOfOuterVertices;
double _numberOfInnerCells;
double _numberOfOuterCells;
double _numberOfInnerLeafVertices;
double _numberOfBoundaryLeafVertices;
double _numberOfOuterLeafVertices;
double _numberOfInnerLeafCells;
double _numberOfOuterLeafCells;
int _maxLevel;
bool _hasRefined;
bool _hasTriggeredRefinementForNextIteration;
bool _hasErased;
bool _hasTriggeredEraseForNextIteration;
bool _hasChangedVertexOrCellState;
bool _hasModifiedGridInPreviousIteration;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;

PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_minMeshWidth = (minMeshWidth);
}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxMeshWidth = (maxMeshWidth);
}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

State();


State(const PersistentRecords& persistentRecords);


State(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~State();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._minMeshWidth = (minMeshWidth);
}



inline double getMinMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._minMeshWidth[elementIndex];

}



inline void setMinMeshWidth(int elementIndex, const double& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._minMeshWidth[elementIndex]= minMeshWidth;

}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxMeshWidth = (maxMeshWidth);
}



inline double getMaxMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._maxMeshWidth[elementIndex];

}



inline void setMaxMeshWidth(int elementIndex, const double& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._maxMeshWidth[elementIndex]= maxMeshWidth;

}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

StatePacked convert() const;


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


class dem::records::StatePacked { 

public:

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
tarch::la::Vector<DIMENSIONS,double> _minMeshWidth;
tarch::la::Vector<DIMENSIONS,double> _maxMeshWidth;
double _numberOfInnerVertices;
double _numberOfBoundaryVertices;
double _numberOfOuterVertices;
double _numberOfInnerCells;
double _numberOfOuterCells;
double _numberOfInnerLeafVertices;
double _numberOfBoundaryLeafVertices;
double _numberOfOuterLeafVertices;
double _numberOfInnerLeafCells;
double _numberOfOuterLeafCells;
int _maxLevel;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_minMeshWidth = (minMeshWidth);
}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxMeshWidth = (maxMeshWidth);
}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( hasRefined ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_packedRecords0 = static_cast<short int>( hasErased ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

StatePacked();


StatePacked(const PersistentRecords& persistentRecords);


StatePacked(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth, const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth, const double& numberOfInnerVertices, const double& numberOfBoundaryVertices, const double& numberOfOuterVertices, const double& numberOfInnerCells, const double& numberOfOuterCells, const double& numberOfInnerLeafVertices, const double& numberOfBoundaryLeafVertices, const double& numberOfOuterLeafVertices, const double& numberOfInnerLeafCells, const double& numberOfOuterLeafCells, const int& maxLevel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~StatePacked();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}




inline tarch::la::Vector<DIMENSIONS,double> getMinMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._minMeshWidth;
}




inline void setMinMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._minMeshWidth = (minMeshWidth);
}



inline double getMinMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._minMeshWidth[elementIndex];

}



inline void setMinMeshWidth(int elementIndex, const double& minMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._minMeshWidth[elementIndex]= minMeshWidth;

}




inline tarch::la::Vector<DIMENSIONS,double> getMaxMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxMeshWidth;
}




inline void setMaxMeshWidth(const tarch::la::Vector<DIMENSIONS,double>& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxMeshWidth = (maxMeshWidth);
}



inline double getMaxMeshWidth(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._maxMeshWidth[elementIndex];

}



inline void setMaxMeshWidth(int elementIndex, const double& maxMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._maxMeshWidth[elementIndex]= maxMeshWidth;

}



inline double getNumberOfInnerVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerVertices;
}



inline void setNumberOfInnerVertices(const double& numberOfInnerVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerVertices = numberOfInnerVertices;
}



inline double getNumberOfBoundaryVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryVertices;
}



inline void setNumberOfBoundaryVertices(const double& numberOfBoundaryVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryVertices = numberOfBoundaryVertices;
}



inline double getNumberOfOuterVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterVertices;
}



inline void setNumberOfOuterVertices(const double& numberOfOuterVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterVertices = numberOfOuterVertices;
}



inline double getNumberOfInnerCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerCells;
}



inline void setNumberOfInnerCells(const double& numberOfInnerCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerCells = numberOfInnerCells;
}



inline double getNumberOfOuterCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterCells;
}



inline void setNumberOfOuterCells(const double& numberOfOuterCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterCells = numberOfOuterCells;
}



inline double getNumberOfInnerLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafVertices;
}



inline void setNumberOfInnerLeafVertices(const double& numberOfInnerLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafVertices = numberOfInnerLeafVertices;
}



inline double getNumberOfBoundaryLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfBoundaryLeafVertices;
}



inline void setNumberOfBoundaryLeafVertices(const double& numberOfBoundaryLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfBoundaryLeafVertices = numberOfBoundaryLeafVertices;
}



inline double getNumberOfOuterLeafVertices() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafVertices;
}



inline void setNumberOfOuterLeafVertices(const double& numberOfOuterLeafVertices) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafVertices = numberOfOuterLeafVertices;
}



inline double getNumberOfInnerLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfInnerLeafCells;
}



inline void setNumberOfInnerLeafCells(const double& numberOfInnerLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfInnerLeafCells = numberOfInnerLeafCells;
}



inline double getNumberOfOuterLeafCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfOuterLeafCells;
}



inline void setNumberOfOuterLeafCells(const double& numberOfOuterLeafCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfOuterLeafCells = numberOfOuterLeafCells;
}



inline int getMaxLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxLevel;
}



inline void setMaxLevel(const int& maxLevel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxLevel = maxLevel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( hasRefined ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_persistentRecords._packedRecords0 = static_cast<short int>( hasErased ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_persistentRecords._packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_persistentRecords._packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

State convert() const;


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



#elif !defined(TrackGridStatistics) && defined(Parallel)

class dem::records::State { 

public:

typedef dem::records::StatePacked Packed;

enum BoundaryRefinement {
RefineArtificially = 0, Nop = 1, EraseAggressively = 2
};

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
bool _hasRefined;
bool _hasTriggeredRefinementForNextIteration;
bool _hasErased;
bool _hasTriggeredEraseForNextIteration;
bool _hasChangedVertexOrCellState;
bool _hasModifiedGridInPreviousIteration;
bool _isTraversalInverted;
bool _reduceStateAndCell;
bool _couldNotEraseDueToDecompositionFlag;
bool _subWorkerIsInvolvedInJoinOrFork;
BoundaryRefinement _refineArtificiallyOutsideDomain;
int _totalNumberOfBatchIterations;
int _batchIteration;

PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _reduceStateAndCell;
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_reduceStateAndCell = reduceStateAndCell;
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _couldNotEraseDueToDecompositionFlag;
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_couldNotEraseDueToDecompositionFlag = couldNotEraseDueToDecompositionFlag;
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subWorkerIsInvolvedInJoinOrFork;
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subWorkerIsInvolvedInJoinOrFork = subWorkerIsInvolvedInJoinOrFork;
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refineArtificiallyOutsideDomain;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refineArtificiallyOutsideDomain = refineArtificiallyOutsideDomain;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

State();


State(const PersistentRecords& persistentRecords);


State(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~State();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasRefined;
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasRefined = hasRefined;
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredRefinementForNextIteration;
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredRefinementForNextIteration = hasTriggeredRefinementForNextIteration;
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasErased;
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasErased = hasErased;
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasTriggeredEraseForNextIteration;
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasTriggeredEraseForNextIteration = hasTriggeredEraseForNextIteration;
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasChangedVertexOrCellState;
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasChangedVertexOrCellState = hasChangedVertexOrCellState;
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._hasModifiedGridInPreviousIteration;
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._hasModifiedGridInPreviousIteration = hasModifiedGridInPreviousIteration;
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._reduceStateAndCell;
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._reduceStateAndCell = reduceStateAndCell;
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._couldNotEraseDueToDecompositionFlag;
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._couldNotEraseDueToDecompositionFlag = couldNotEraseDueToDecompositionFlag;
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subWorkerIsInvolvedInJoinOrFork;
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subWorkerIsInvolvedInJoinOrFork = subWorkerIsInvolvedInJoinOrFork;
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refineArtificiallyOutsideDomain;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refineArtificiallyOutsideDomain = refineArtificiallyOutsideDomain;
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



static std::string toString(const BoundaryRefinement& param);


static std::string getBoundaryRefinementMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

StatePacked convert() const;


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


class dem::records::StatePacked { 

public:

typedef dem::records::State::BoundaryRefinement BoundaryRefinement;

struct PersistentRecords {
double _numberOfContactPoints;
double _numberOfParticleReassignments;
double _numberOfTriangleComparisons;
double _numberOfParticleComparisons;
bool _adaptiveStepSize;
double _timeStepSize;
int _timeStep;
double _currentTime;
double _stepIncrement;
double _twoParticlesAreClose;
bool _twoParticlesSeparate;
int _numberOfParticles;
int _numberOfObstacles;
double _prescribedMinimumMeshWidth;
double _prescribedMaximumMeshWidth;
double _maxVelocityApproach;
double _maxVelocityTravel;
bool _isTraversalInverted;
int _totalNumberOfBatchIterations;
int _batchIteration;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( hasRefined ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_packedRecords0 = static_cast<short int>( hasErased ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( reduceStateAndCell ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( couldNotEraseDueToDecompositionFlag ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_packedRecords0 = static_cast<short int>( subWorkerIsInvolvedInJoinOrFork ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (9));
assertion(( tmp >= 0 &&  tmp <= 2));
return (BoundaryRefinement) tmp;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refineArtificiallyOutsideDomain >= 0 && refineArtificiallyOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refineArtificiallyOutsideDomain) << (9));
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_batchIteration = batchIteration;
}



};
private: 
PersistentRecords _persistentRecords;

public:

StatePacked();


StatePacked(const PersistentRecords& persistentRecords);


StatePacked(const double& numberOfContactPoints, const double& numberOfParticleReassignments, const double& numberOfTriangleComparisons, const double& numberOfParticleComparisons, const bool& adaptiveStepSize, const double& timeStepSize, const int& timeStep, const double& currentTime, const double& stepIncrement, const double& twoParticlesAreClose, const bool& twoParticlesSeparate, const int& numberOfParticles, const int& numberOfObstacles, const double& prescribedMinimumMeshWidth, const double& prescribedMaximumMeshWidth, const double& maxVelocityApproach, const double& maxVelocityTravel, const bool& hasRefined, const bool& hasTriggeredRefinementForNextIteration, const bool& hasErased, const bool& hasTriggeredEraseForNextIteration, const bool& hasChangedVertexOrCellState, const bool& hasModifiedGridInPreviousIteration, const bool& isTraversalInverted, const bool& reduceStateAndCell, const bool& couldNotEraseDueToDecompositionFlag, const bool& subWorkerIsInvolvedInJoinOrFork, const BoundaryRefinement& refineArtificiallyOutsideDomain, const int& totalNumberOfBatchIterations, const int& batchIteration);


virtual ~StatePacked();


inline double getNumberOfContactPoints() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfContactPoints;
}



inline void setNumberOfContactPoints(const double& numberOfContactPoints) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfContactPoints = numberOfContactPoints;
}



inline double getNumberOfParticleReassignments() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleReassignments;
}



inline void setNumberOfParticleReassignments(const double& numberOfParticleReassignments) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleReassignments = numberOfParticleReassignments;
}



inline double getNumberOfTriangleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfTriangleComparisons;
}



inline void setNumberOfTriangleComparisons(const double& numberOfTriangleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfTriangleComparisons = numberOfTriangleComparisons;
}



inline double getNumberOfParticleComparisons() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticleComparisons;
}



inline void setNumberOfParticleComparisons(const double& numberOfParticleComparisons) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticleComparisons = numberOfParticleComparisons;
}



inline bool getAdaptiveStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adaptiveStepSize;
}



inline void setAdaptiveStepSize(const bool& adaptiveStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adaptiveStepSize = adaptiveStepSize;
}



inline double getTimeStepSize() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStepSize;
}



inline void setTimeStepSize(const double& timeStepSize) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStepSize = timeStepSize;
}



inline int getTimeStep() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._timeStep;
}



inline void setTimeStep(const int& timeStep) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._timeStep = timeStep;
}



inline double getCurrentTime() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._currentTime;
}



inline void setCurrentTime(const double& currentTime) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._currentTime = currentTime;
}



inline double getStepIncrement() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._stepIncrement;
}



inline void setStepIncrement(const double& stepIncrement) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._stepIncrement = stepIncrement;
}



inline double getTwoParticlesAreClose() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesAreClose;
}



inline void setTwoParticlesAreClose(const double& twoParticlesAreClose) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesAreClose = twoParticlesAreClose;
}



inline bool getTwoParticlesSeparate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._twoParticlesSeparate;
}



inline void setTwoParticlesSeparate(const bool& twoParticlesSeparate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._twoParticlesSeparate = twoParticlesSeparate;
}



inline int getNumberOfParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticles;
}



inline void setNumberOfParticles(const int& numberOfParticles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticles = numberOfParticles;
}



inline int getNumberOfObstacles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfObstacles;
}



inline void setNumberOfObstacles(const int& numberOfObstacles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfObstacles = numberOfObstacles;
}



inline double getPrescribedMinimumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMinimumMeshWidth;
}



inline void setPrescribedMinimumMeshWidth(const double& prescribedMinimumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMinimumMeshWidth = prescribedMinimumMeshWidth;
}



inline double getPrescribedMaximumMeshWidth() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._prescribedMaximumMeshWidth;
}



inline void setPrescribedMaximumMeshWidth(const double& prescribedMaximumMeshWidth) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._prescribedMaximumMeshWidth = prescribedMaximumMeshWidth;
}



inline double getMaxVelocityApproach() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityApproach;
}



inline void setMaxVelocityApproach(const double& maxVelocityApproach) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityApproach = maxVelocityApproach;
}



inline double getMaxVelocityTravel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._maxVelocityTravel;
}



inline void setMaxVelocityTravel(const double& maxVelocityTravel) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._maxVelocityTravel = maxVelocityTravel;
}



inline bool getHasRefined() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasRefined(const bool& hasRefined) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( hasRefined ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredRefinementForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredRefinementForNextIteration(const bool& hasTriggeredRefinementForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (1);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredRefinementForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasErased() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasErased(const bool& hasErased) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (2);
_persistentRecords._packedRecords0 = static_cast<short int>( hasErased ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasTriggeredEraseForNextIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasTriggeredEraseForNextIteration(const bool& hasTriggeredEraseForNextIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (3);
_persistentRecords._packedRecords0 = static_cast<short int>( hasTriggeredEraseForNextIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasChangedVertexOrCellState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasChangedVertexOrCellState(const bool& hasChangedVertexOrCellState) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (4);
_persistentRecords._packedRecords0 = static_cast<short int>( hasChangedVertexOrCellState ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getHasModifiedGridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setHasModifiedGridInPreviousIteration(const bool& hasModifiedGridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (5);
_persistentRecords._packedRecords0 = static_cast<short int>( hasModifiedGridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getIsTraversalInverted() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isTraversalInverted;
}



inline void setIsTraversalInverted(const bool& isTraversalInverted) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isTraversalInverted = isTraversalInverted;
}



inline bool getReduceStateAndCell() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setReduceStateAndCell(const bool& reduceStateAndCell) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( reduceStateAndCell ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getCouldNotEraseDueToDecompositionFlag() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCouldNotEraseDueToDecompositionFlag(const bool& couldNotEraseDueToDecompositionFlag) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( couldNotEraseDueToDecompositionFlag ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getSubWorkerIsInvolvedInJoinOrFork() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setSubWorkerIsInvolvedInJoinOrFork(const bool& subWorkerIsInvolvedInJoinOrFork) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_persistentRecords._packedRecords0 = static_cast<short int>( subWorkerIsInvolvedInJoinOrFork ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline BoundaryRefinement getRefineArtificiallyOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (9));
assertion(( tmp >= 0 &&  tmp <= 2));
return (BoundaryRefinement) tmp;
}



inline void setRefineArtificiallyOutsideDomain(const BoundaryRefinement& refineArtificiallyOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refineArtificiallyOutsideDomain >= 0 && refineArtificiallyOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (9));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refineArtificiallyOutsideDomain) << (9));
}



inline int getTotalNumberOfBatchIterations() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._totalNumberOfBatchIterations;
}



inline void setTotalNumberOfBatchIterations(const int& totalNumberOfBatchIterations) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._totalNumberOfBatchIterations = totalNumberOfBatchIterations;
}



inline int getBatchIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._batchIteration;
}



inline void setBatchIteration(const int& batchIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._batchIteration = batchIteration;
}



static std::string toString(const BoundaryRefinement& param);


static std::string getBoundaryRefinementMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

State convert() const;


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

#endif

