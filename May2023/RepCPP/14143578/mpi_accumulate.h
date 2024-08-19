
#pragma once
#include <mpi.h>
#include <array>
#include <vector>
#include <map>
#include "accumulate.h"

namespace dg
{
namespace exblas {

namespace detail{
static std::map<MPI_Comm, std::array<MPI_Comm, 2>> comm_mods;
}

static void mpi_reduce_communicator(MPI_Comm comm, MPI_Comm* comm_mod, MPI_Comm* comm_mod_reduce){
assert( comm != MPI_COMM_NULL);
if( detail::comm_mods.count(comm) == 1 )
{
*comm_mod = detail::comm_mods[comm][0];
*comm_mod_reduce = detail::comm_mods[comm][1];
return;
}
else
{
int mod = 128;
int rank, size;
MPI_Comm_rank( comm, &rank);
MPI_Comm_size( comm, &size);
MPI_Comm_split( comm, rank/mod, rank%mod, comm_mod); 
MPI_Group group, reduce_group;
MPI_Comm_group( comm, &group); 
int reduce_size=(int)ceil((double)size/(double)mod);
std::vector<int> reduce_ranks(reduce_size);
for( int i=0; i<reduce_size; i++)
reduce_ranks[i] = i*mod;
MPI_Group_incl( group, reduce_size, reduce_ranks.data(), &reduce_group); 
MPI_Comm_create( comm, reduce_group, comm_mod_reduce); 
MPI_Group_free( &group);
MPI_Group_free( &reduce_group);
detail::comm_mods[comm] = {*comm_mod, *comm_mod_reduce};
}
}


static void reduce_mpi_cpu(  unsigned num_superacc, int64_t* in, int64_t* out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce )
{
for( unsigned i=0; i<num_superacc; i++)
{
int imin=exblas::IMIN, imax=exblas::IMAX;
cpu::Normalize(&in[i*exblas::BIN_COUNT], imin, imax);
}
MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod);
int rank;
MPI_Comm_rank( comm_mod, &rank);
if(comm_mod_reduce != MPI_COMM_NULL)
{
for( unsigned i=0; i<num_superacc; i++)
{
int imin=exblas::IMIN, imax=exblas::IMAX;
cpu::Normalize(&out[i*exblas::BIN_COUNT], imin, imax);
for( int k=0; k<exblas::BIN_COUNT; k++)
in[i*BIN_COUNT+k] = out[i*BIN_COUNT+k];
}
MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod_reduce);
}
MPI_Bcast( out, num_superacc*exblas::BIN_COUNT, MPI_LONG, 0, comm);
}

}
} 
