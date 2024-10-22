

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_ibarrier_(MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err);

void mpi_barrier_(MPI_Fint *comm, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
MPI_Fint request;
mpi_ibarrier_(comm, &request, err);
if (*err == MPI_SUCCESS)
RequestManager<Fortran>::processRequest(request);
} else {
static mpi_barrier_t *symbol = (mpi_barrier_t *) Symbol::load(__func__);
(*symbol)(comm, err);
}
}

void tampi_ibarrier_internal_(MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err)
{
mpi_ibarrier_(comm, request, err);

if (Environment::isNonBlockingEnabled()) {
if (*err == MPI_SUCCESS) {
tampi_iwait_(request, MPI_F_STATUS_IGNORE, err);
}
}
}
} 

#pragma GCC visibility pop
