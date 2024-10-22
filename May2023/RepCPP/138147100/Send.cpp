

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_isend_(void *buf, MPI_Fint *count, MPI_Fint *datatype,
MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
MPI_Fint *request, MPI_Fint *err);

void mpi_send_(void *buf, MPI_Fint *count, MPI_Fint *datatype,
MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
MPI_Fint request;
mpi_isend_(buf, count, datatype, dest, tag, comm, &request, err);
if (*err == MPI_SUCCESS)
RequestManager<Fortran>::processRequest(request);
} else {
static mpi_send_t *symbol = (mpi_send_t *) Symbol::load(__func__);
(*symbol)(buf, count, datatype, dest, tag, comm, err);
}
}

void tampi_isend_internal_(void *buf, MPI_Fint *count, MPI_Fint *datatype,
MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm,
MPI_Fint *request, MPI_Fint *err)
{
mpi_isend_(buf, count, datatype, dest, tag, comm, request, err);

if (Environment::isNonBlockingEnabled()) {
if (*err == MPI_SUCCESS) {
tampi_iwait_(request, MPI_F_STATUS_IGNORE, err);
}
}
}
} 

#pragma GCC visibility pop
