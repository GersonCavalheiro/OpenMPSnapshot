

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
void tampi_iwait_(MPI_Fint *request, MPI_Fint *status, MPI_Fint *err)
{
if (Environment::isNonBlockingEnabled()) {
RequestManager<Fortran>::processRequest(*request, status,  false);
}
*err = MPI_SUCCESS;
}
} 

#pragma GCC visibility pop
