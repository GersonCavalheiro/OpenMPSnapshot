#pragma once

#ifndef HAS_NO_MPI

#include <mpi.h> 
auto const MPI_UINT16 = MPI_UNSIGNED_SHORT;

#else 


#include <cassert> 
#include <cstdint> 
#include <cstring> 

#include "recorded_warnings.hxx" 

typedef int64_t MPI_Comm;
MPI_Comm constexpr MPI_COMM_WORLD = -1;
int constexpr MPI_SUCCESS = 0;

typedef char MPI_Op;
MPI_Op constexpr MPI_SUM = '+', MPI_MAX = 'M', MPI_MIN = 'm', MPI_OP_NULL = 0;

typedef int MPI_Datatype;
MPI_Datatype constexpr MPI_UINT16 = 2;
MPI_Datatype constexpr MPI_DOUBLE = -8;
MPI_Datatype constexpr MPI_UNSIGNED_LONG = 8;
void const * const MPI_IN_PLACE = nullptr;

inline size_t const size_of(MPI_Datatype const datatype) {
switch (datatype) {
case 0:          return 1; 
case MPI_UINT16: return 2; 
case MPI_DOUBLE: return 8; 
case MPI_UNSIGNED_LONG: return 8; 
}
warn("unknown MPI_Datatype %d", int(datatype));
return 0;
} 

#define ok   return MPI_SUCCESS
inline int MPI_Init(int *argc, char ***argv) { ok; }
inline int MPI_Finalize(void) { ok; }
inline int MPI_Comm_rank(MPI_Comm comm, int *rank) { assert(rank); *rank = 0; ok; }
inline int MPI_Comm_size(MPI_Comm comm, int *size) { assert(size); *size = 1; ok; }
inline int MPI_Allreduce(void const *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
if (sendbuf) { std::memcpy(recvbuf, sendbuf, count*size_of(datatype)); } ok; }
inline int MPI_Barrier(MPI_Comm comm) { ok; }
inline double MPI_Wtime(void) { return 0; } 
#undef  ok

#endif 

#include <cstdio> 
#include "simple_stats.hxx" 
#include "status.hxx" 

namespace mpi_parallel {

#define MPI_Check(MPI_Function) \
mpi_parallel::__check_MPI_call((MPI_Function), __FILE__, __LINE__, #MPI_Function)

inline int __check_MPI_call( 
int const MPI_status
, char const *const file
, unsigned const line
, char const* const name
) {
#ifdef FULLDEBUG
std::printf("# calling %s in %s:%d returned status= %i\n", name, file, line, MPI_status);
#endif
if (MPI_SUCCESS != MPI_status) {
std::printf("\n# in %s:%d failed with status= %i calling %s\n", file, line, MPI_status, name);
} 
return MPI_status; 
} 




inline int init(int argc=0, char **argv=nullptr) { 
static bool already{false};
if (already) return 1; 
already = true;
return MPI_Check(MPI_Init(&argc, &argv));
} 

inline MPI_Comm comm() { return MPI_COMM_WORLD; }

inline unsigned size(MPI_Comm const comm=MPI_COMM_WORLD) {
int size{0};
MPI_Check(MPI_Comm_size(comm, &size));
assert( size > 0 );
return size;
} 

inline int rank(MPI_Comm const comm=MPI_COMM_WORLD, unsigned const check_size=0) { 
int rank{-1};
MPI_Check(MPI_Comm_rank(comm, &rank));
assert( rank >= 0 );
if (check_size > 0) assert( rank < check_size );
return rank;
} 

template <typename T> MPI_Datatype get(T t=0);
template <> inline MPI_Datatype get<uint16_t>(uint16_t t) { return MPI_UINT16; }
template <> inline MPI_Datatype get<double>(double t) { return MPI_DOUBLE; }
template <> inline MPI_Datatype get<size_t>(size_t t) { return MPI_UNSIGNED_LONG; }

template <typename T>
inline int allreduce(T *recv, MPI_Op const op=MPI_SUM, MPI_Comm const comm=MPI_COMM_WORLD, size_t const count=1, T const *send=nullptr) {
if (!send) send = (T const *)MPI_IN_PLACE;
return MPI_Allreduce(send, recv, count, get<T>(), op, comm);
} 

template <typename T>
inline int max(T *recv, size_t const count=1, MPI_Comm const comm=MPI_COMM_WORLD) {
return allreduce(recv, MPI_MAX, comm, count); }

template <typename T>
inline int sum(T *recv, size_t const count=1, MPI_Comm const comm=MPI_COMM_WORLD) {
return allreduce(recv, MPI_SUM, comm, count); }

inline int barrier(MPI_Comm const comm=MPI_COMM_WORLD) { 
return MPI_Check(MPI_Barrier(comm));
} 

inline int finalize(void) {
return MPI_Check(MPI_Finalize());
} 


inline int allreduce(simple_stats::Stats<double> & stats, MPI_Comm const comm=MPI_COMM_WORLD) {
double v[8];
stats.get(v);
auto const status_sum = sum(v, 5, comm);
auto const status_max = max(v + 6, 2, comm); 
stats.set(v);
return status_sum + status_max;
} 


status_t all_tests(int const echo=0); 

} 
