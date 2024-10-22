

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_wait_(MPI_Fint *request, MPI_Fint *status, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
RequestManager<Fortran>::processRequest(*request, status);
} else {
static mpi_wait_t *symbol = (mpi_wait_t *) Symbol::load(__func__);
(*symbol)(request, status, err);
}
}

void tampi_wait_internal_(MPI_Fint *request, MPI_Fint *status, MPI_Fint *err)
{
*err = MPI_SUCCESS;
if (!Environment::isNonBlockingEnabled()) {
mpi_wait_(request, status, err);
}
}
} 

#pragma GCC visibility pop
