#ifndef _DEM_RECORDS_VERTEX_H
#define _DEM_RECORDS_VERTEX_H

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
class Vertex;
class VertexPacked;
}
}

#if defined(Parallel) && defined(PersistentRegularSubtrees) && defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5, RefineDueToJoinThoughWorkerIsAlreadyErasing = 6, EnforceRefinementTriggered = 7
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _x __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _x;
#endif
int _level;
#ifdef UseManualAlignment
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;
#endif
bool _adjacentSubtreeForksIntoOtherRank;
bool _parentRegularPersistentSubgrid;
bool _parentRegularPersistentSubgridInPreviousIteration;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<DIMENSIONS,double> _x;
int _level;
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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


#elif defined(PersistentRegularSubtrees) && defined(Asserts) && !defined(Parallel)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _x __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _x;
#endif
int _level;
bool _parentRegularPersistentSubgrid;
bool _parentRegularPersistentSubgridInPreviousIteration;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<DIMENSIONS,double> _x;
int _level;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5, RefineDueToJoinThoughWorkerIsAlreadyErasing = 6, EnforceRefinementTriggered = 7
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _x __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _x;
#endif
int _level;
#ifdef UseManualAlignment
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;
#endif
bool _adjacentSubtreeForksIntoOtherRank;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<DIMENSIONS,double> _x;
int _level;
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif defined(Parallel) && defined(PersistentRegularSubtrees) && !defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5, RefineDueToJoinThoughWorkerIsAlreadyErasing = 6, EnforceRefinementTriggered = 7
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;
#endif
bool _adjacentSubtreeForksIntoOtherRank;
bool _parentRegularPersistentSubgrid;
bool _parentRegularPersistentSubgridInPreviousIteration;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (8);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
bool _parentRegularPersistentSubgrid;
bool _parentRegularPersistentSubgridInPreviousIteration;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgrid;
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgrid = parentRegularPersistentSubgrid;
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._parentRegularPersistentSubgridInPreviousIteration;
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._parentRegularPersistentSubgridInPreviousIteration = parentRegularPersistentSubgridInPreviousIteration;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const bool& parentRegularPersistentSubgrid, const bool& parentRegularPersistentSubgridInPreviousIteration);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}



inline bool getParentRegularPersistentSubgrid() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgrid(const bool& parentRegularPersistentSubgrid) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgrid ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline bool getParentRegularPersistentSubgridInPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setParentRegularPersistentSubgridInPreviousIteration(const bool& parentRegularPersistentSubgridInPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (7);
_persistentRecords._packedRecords0 = static_cast<short int>( parentRegularPersistentSubgridInPreviousIteration ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && !defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5, RefineDueToJoinThoughWorkerIsAlreadyErasing = 6, EnforceRefinementTriggered = 7
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;
#endif
bool _adjacentSubtreeForksIntoOtherRank;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentSubtreeForksIntoOtherRank;
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentSubtreeForksIntoOtherRank = adjacentSubtreeForksIntoOtherRank;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<TWO_POWER_D,int> _adjacentRanks;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentRanks = (adjacentRanks);
}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks, const bool& adjacentSubtreeForksIntoOtherRank);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 7));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 7));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<TWO_POWER_D,int> getAdjacentRanks() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentRanks;
}




inline void setAdjacentRanks(const tarch::la::Vector<TWO_POWER_D,int>& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentRanks = (adjacentRanks);
}



inline int getAdjacentRanks(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
return _persistentRecords._adjacentRanks[elementIndex];

}



inline void setAdjacentRanks(int elementIndex, const int& adjacentRanks) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<TWO_POWER_D);
_persistentRecords._adjacentRanks[elementIndex]= adjacentRanks;

}



inline bool getAdjacentSubtreeForksIntoOtherRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setAdjacentSubtreeForksIntoOtherRank(const bool& adjacentSubtreeForksIntoOtherRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (6);
_persistentRecords._packedRecords0 = static_cast<short int>( adjacentSubtreeForksIntoOtherRank ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif !defined(PersistentRegularSubtrees) && defined(Asserts) && !defined(Parallel)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS,double> _x __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS,double> _x;
#endif
int _level;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;
tarch::la::Vector<DIMENSIONS,double> _x;
int _level;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_x = (x);
}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_level = level;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain, const tarch::la::Vector<DIMENSIONS,double>& x, const int& level);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}




inline tarch::la::Vector<DIMENSIONS,double> getX() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._x;
}




inline void setX(const tarch::la::Vector<DIMENSIONS,double>& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._x = (x);
}



inline double getX(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._x[elementIndex];

}



inline void setX(int elementIndex, const double& x) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._x[elementIndex]= x;

}



inline int getLevel() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._level;
}



inline void setLevel(const int& level) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._level = level;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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



#elif !defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Asserts)

class dem::records::Vertex { 

public:

typedef dem::records::VertexPacked Packed;

enum InsideOutsideDomain {
Inside = 0, Boundary = 1, Outside = 2
};

enum RefinementControl {
Unrefined = 0, Refined = 1, RefinementTriggered = 2, Refining = 3, EraseTriggered = 4, Erasing = 5
};

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
bool _isHangingNode;
RefinementControl _refinementControl;
int _adjacentCellsHeight;
InsideOutsideDomain _insideOutsideDomain;

PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_insideOutsideDomain = insideOutsideDomain;
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

Vertex();


Vertex(const PersistentRecords& persistentRecords);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain);


Vertex(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain);


virtual ~Vertex();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isHangingNode;
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isHangingNode = isHangingNode;
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._refinementControl;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._refinementControl = refinementControl;
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._insideOutsideDomain;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._insideOutsideDomain = insideOutsideDomain;
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

VertexPacked convert() const;


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


class dem::records::VertexPacked { 

public:

typedef dem::records::Vertex::InsideOutsideDomain InsideOutsideDomain;

typedef dem::records::Vertex::RefinementControl RefinementControl;

struct PersistentRecords {
int _particles;
int _particlesOnCoarserLevels;
double _numberOfParticlesInUnrefinedVertex;
int _adjacentCellsHeight;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain);


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isHangingNode ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeight = adjacentCellsHeight;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}



};
private: 
PersistentRecords _persistentRecords;
int _adjacentCellsHeightOfPreviousIteration;
int _numberOfAdjacentRefinedCells;

public:

VertexPacked();


VertexPacked(const PersistentRecords& persistentRecords);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const InsideOutsideDomain& insideOutsideDomain);


VertexPacked(const int& particles, const int& particlesOnCoarserLevels, const double& numberOfParticlesInUnrefinedVertex, const bool& isHangingNode, const RefinementControl& refinementControl, const int& adjacentCellsHeight, const int& adjacentCellsHeightOfPreviousIteration, const int& numberOfAdjacentRefinedCells, const InsideOutsideDomain& insideOutsideDomain);


virtual ~VertexPacked();


inline int getParticles() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particles;
}



inline void setParticles(const int& particles) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particles = particles;
}



inline int getParticlesOnCoarserLevels() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._particlesOnCoarserLevels;
}



inline void setParticlesOnCoarserLevels(const int& particlesOnCoarserLevels) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._particlesOnCoarserLevels = particlesOnCoarserLevels;
}



inline double getNumberOfParticlesInUnrefinedVertex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfParticlesInUnrefinedVertex;
}



inline void setNumberOfParticlesInUnrefinedVertex(const double& numberOfParticlesInUnrefinedVertex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfParticlesInUnrefinedVertex = numberOfParticlesInUnrefinedVertex;
}



inline bool getIsHangingNode() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsHangingNode(const bool& isHangingNode) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isHangingNode ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline RefinementControl getRefinementControl() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 5));
return (RefinementControl) tmp;
}



inline void setRefinementControl(const RefinementControl& refinementControl) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((refinementControl >= 0 && refinementControl <= 5));
short int mask =  (1 << (3)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(refinementControl) << (1));
}



inline int getAdjacentCellsHeight() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._adjacentCellsHeight;
}



inline void setAdjacentCellsHeight(const int& adjacentCellsHeight) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._adjacentCellsHeight = adjacentCellsHeight;
}



inline int getAdjacentCellsHeightOfPreviousIteration() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _adjacentCellsHeightOfPreviousIteration;
}



inline void setAdjacentCellsHeightOfPreviousIteration(const int& adjacentCellsHeightOfPreviousIteration) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_adjacentCellsHeightOfPreviousIteration = adjacentCellsHeightOfPreviousIteration;
}



inline int getNumberOfAdjacentRefinedCells() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfAdjacentRefinedCells;
}



inline void setNumberOfAdjacentRefinedCells(const int& numberOfAdjacentRefinedCells) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfAdjacentRefinedCells = numberOfAdjacentRefinedCells;
}



inline InsideOutsideDomain getInsideOutsideDomain() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (4));
assertion(( tmp >= 0 &&  tmp <= 2));
return (InsideOutsideDomain) tmp;
}



inline void setInsideOutsideDomain(const InsideOutsideDomain& insideOutsideDomain) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((insideOutsideDomain >= 0 && insideOutsideDomain <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (4));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(insideOutsideDomain) << (4));
}



static std::string toString(const InsideOutsideDomain& param);


static std::string getInsideOutsideDomainMapping();


static std::string toString(const RefinementControl& param);


static std::string getRefinementControlMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Vertex convert() const;


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

