#ifndef _DEM_RECORDS_CELL_H
#define _DEM_RECORDS_CELL_H

#include "tarch/multicore/MulticoreDefinitions.h"
#include "peano/utils/PeanoOptimisations.h"
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
class Cell;
class CellPacked;
}
}

#if !defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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


#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && !defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif !defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif !defined(PersistentRegularSubtrees) && defined(Debug) && !defined(Parallel) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(PersistentRegularSubtrees) && !defined(Parallel) && !defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif !defined(PersistentRegularSubtrees) && defined(Debug) && !defined(Parallel) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(PersistentRegularSubtrees) && defined(Debug) && !defined(Parallel) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && !defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && defined(PersistentRegularSubtrees) && !defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(PersistentRegularSubtrees) && defined(Debug) && !defined(Parallel) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && !defined(PersistentRegularSubtrees) && defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && defined(PersistentRegularSubtrees) && !defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
}




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && defined(PersistentRegularSubtrees) && defined(Debug) && !defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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



#elif defined(Parallel) && defined(PersistentRegularSubtrees) && defined(Debug) && defined(SharedMemoryParallelisation)

class dem::records::Cell { 

public:

typedef dem::records::CellPacked Packed;

enum State {
Leaf = 0, Refined = 1, Root = 2
};

struct PersistentRecords {
bool _isInside;
State _state;
int _level;
#ifdef UseManualAlignment
std::bitset<DIMENSIONS> _evenFlags __attribute__((aligned(VectorisationAlignment)));
#else
std::bitset<DIMENSIONS> _evenFlags;
#endif
#ifdef UseManualAlignment
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber __attribute__((aligned(VectorisationAlignment)));
#else
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
#endif
int _responsibleRank;
bool _subtreeHoldsWorker;
bool _cellIsAForkCandidate;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;

PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_evenFlags = (evenFlags);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

Cell();


Cell(const PersistentRecords& persistentRecords);


Cell(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~Cell();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._isInside;
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._isInside = isInside;
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._state;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._state = state;
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._evenFlags;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._evenFlags = (evenFlags);
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
return _persistentRecords._evenFlags[elementIndex];

}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags[elementIndex]= evenFlags;

}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
_persistentRecords._evenFlags.flip(elementIndex);
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._cellIsAForkCandidate;
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._cellIsAForkCandidate = cellIsAForkCandidate;
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

CellPacked convert() const;


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


class dem::records::CellPacked { 

public:

typedef dem::records::Cell::State State;

struct PersistentRecords {
int _level;
tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> _accessNumber;
int _responsibleRank;
bool _subtreeHoldsWorker;
int _numberOfLoadsFromInputStream;
int _numberOfStoresToOutputStream;
int _persistentRegularSubtreeIndex;


short int _packedRecords0;


PersistentRecords();


PersistentRecords(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_packedRecords0 = static_cast<short int>( isInside ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_packedRecords0 = static_cast<short int>(_packedRecords0 & ~mask);
_packedRecords0 = static_cast<short int>(_packedRecords0 | evenFlags.to_ulong() << (3));
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_accessNumber = (accessNumber);
}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_packedRecords0 | mask) : (_packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



};
private: 
PersistentRecords _persistentRecords;

public:

CellPacked();


CellPacked(const PersistentRecords& persistentRecords);


CellPacked(const bool& isInside, const State& state, const int& level, const std::bitset<DIMENSIONS>& evenFlags, const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber, const int& responsibleRank, const bool& subtreeHoldsWorker, const bool& cellIsAForkCandidate, const int& numberOfLoadsFromInputStream, const int& numberOfStoresToOutputStream, const int& persistentRegularSubtreeIndex);


virtual ~CellPacked();


inline bool getIsInside() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setIsInside(const bool& isInside) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (0);
_persistentRecords._packedRecords0 = static_cast<short int>( isInside ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline State getState() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (1));
assertion(( tmp >= 0 &&  tmp <= 2));
return (State) tmp;
}



inline void setState(const State& state) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion((state >= 0 && state <= 2));
short int mask =  (1 << (2)) - 1;
mask = static_cast<short int>(mask << (1));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | static_cast<short int>(state) << (1));
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




inline std::bitset<DIMENSIONS> getEvenFlags() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
tmp = static_cast<short int>(tmp >> (3));
std::bitset<DIMENSIONS> result = tmp;
return result;
}




inline void setEvenFlags(const std::bitset<DIMENSIONS>& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = (short int) (1 << (DIMENSIONS)) - 1 ;
mask = static_cast<short int>(mask << (3));
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 & ~mask);
_persistentRecords._packedRecords0 = static_cast<short int>(_persistentRecords._packedRecords0 | evenFlags.to_ulong() << (3));
}



inline bool getEvenFlags(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
int mask = 1 << (3);
mask = mask << elementIndex;
return (_persistentRecords._packedRecords0& mask);
}



inline void setEvenFlags(int elementIndex, const bool& evenFlags) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
assertion(!evenFlags || evenFlags==1);
int shift        = 3 + elementIndex; 
short int mask         = 1     << (shift);
short int shiftedValue = evenFlags << (shift);
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 & ~mask;
_persistentRecords._packedRecords0 = _persistentRecords._packedRecords0 |  shiftedValue;
}



inline void flipEvenFlags(int elementIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS);
short int mask = 1 << (3);
mask = mask << elementIndex;
_persistentRecords._packedRecords0^= mask;
}




inline tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int> getAccessNumber() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._accessNumber;
}




inline void setAccessNumber(const tarch::la::Vector<DIMENSIONS_TIMES_TWO,short int>& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._accessNumber = (accessNumber);
}



inline short int getAccessNumber(int elementIndex) const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
return _persistentRecords._accessNumber[elementIndex];

}



inline void setAccessNumber(int elementIndex, const short int& accessNumber) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
assertion(elementIndex>=0);
assertion(elementIndex<DIMENSIONS_TIMES_TWO);
_persistentRecords._accessNumber[elementIndex]= accessNumber;

}



inline int getResponsibleRank() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._responsibleRank;
}



inline void setResponsibleRank(const int& responsibleRank) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._responsibleRank = responsibleRank;
}



inline bool getSubtreeHoldsWorker() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._subtreeHoldsWorker;
}



inline void setSubtreeHoldsWorker(const bool& subtreeHoldsWorker) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._subtreeHoldsWorker = subtreeHoldsWorker;
}



inline bool getCellIsAForkCandidate() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
short int tmp = static_cast<short int>(_persistentRecords._packedRecords0 & mask);
return (tmp != 0);
}



inline void setCellIsAForkCandidate(const bool& cellIsAForkCandidate) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
short int mask = 1 << (DIMENSIONS + 3);
_persistentRecords._packedRecords0 = static_cast<short int>( cellIsAForkCandidate ? (_persistentRecords._packedRecords0 | mask) : (_persistentRecords._packedRecords0 & ~mask));
}



inline int getNumberOfLoadsFromInputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfLoadsFromInputStream;
}



inline void setNumberOfLoadsFromInputStream(const int& numberOfLoadsFromInputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfLoadsFromInputStream = numberOfLoadsFromInputStream;
}



inline int getNumberOfStoresToOutputStream() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._numberOfStoresToOutputStream;
}



inline void setNumberOfStoresToOutputStream(const int& numberOfStoresToOutputStream) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._numberOfStoresToOutputStream = numberOfStoresToOutputStream;
}



inline int getPersistentRegularSubtreeIndex() const 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
return _persistentRecords._persistentRegularSubtreeIndex;
}



inline void setPersistentRegularSubtreeIndex(const int& persistentRegularSubtreeIndex) 
#ifdef UseManualInlining
__attribute__((always_inline))
#endif 
{
_persistentRecords._persistentRegularSubtreeIndex = persistentRegularSubtreeIndex;
}



static std::string toString(const State& param);


static std::string getStateMapping();


std::string toString() const;


void toString(std::ostream& out) const;


PersistentRecords getPersistentRecords() const;

Cell convert() const;


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

