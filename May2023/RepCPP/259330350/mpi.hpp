#pragma once

#include <parthenon/parthenon.hpp>

#ifdef MPI_PARALLEL

#include <mpi.h>

static auto comm = MPI_COMM_WORLD;

inline bool MPIRank()
{
return parthenon::Globals::my_rank;
}
inline bool MPIRank0()
{
return (parthenon::Globals::my_rank == 0 ? true : false);
}
inline void MPIBarrier()
{
MPI_Barrier(comm);
}


template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
static parthenon::AllReduce<T> reduction;
reduction.val = f;
reduction.StartReduce(O);
while (reduction.CheckReduce() == parthenon::TaskStatus::incomplete);
return reduction.val;
}
#else

inline void MPIBarrier() {}
inline bool MPIRank() { return 0; }
inline bool MPIRank0() { return true; }

template<typename T>
inline T MPIReduce_once(T f, MPI_Op O)
{
return f;
}

#endif 
