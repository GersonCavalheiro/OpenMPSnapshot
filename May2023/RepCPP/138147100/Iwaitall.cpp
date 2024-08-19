

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"
#include "util/ErrorHandler.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void tampi_iwaitall_(MPI_Fint *count, MPI_Fint requests[], MPI_Fint *statuses, MPI_Fint *err)
{
if (Environment::isNonBlockingEnabled()) {
RequestManager<Fortran>::processRequests({requests, *count}, statuses,  false);
}
*err = MPI_SUCCESS;
}
} 

#pragma GCC visibility pop
