

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_query_thread_(MPI_Fint *provided, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
*provided = MPI_TASK_MULTIPLE;
*err = MPI_SUCCESS;
} else {
static mpi_query_thread_t *symbol = (mpi_query_thread_t *) Symbol::load(__func__);
(*symbol)(provided, err);
}
}

void tampi_blocking_enabled_(MPI_Fint *flag, MPI_Fint *err)
{
assert(flag != NULL);
*flag = Environment::isBlockingEnabled() ? 1 : 0;
*err = MPI_SUCCESS;
}

void tampi_nonblocking_enabled_(MPI_Fint *flag, MPI_Fint *err)
{
assert(flag != NULL);
*flag = Environment::isNonBlockingEnabled() ? 1 : 0;
*err = MPI_SUCCESS;
}
} 

#pragma GCC visibility pop
