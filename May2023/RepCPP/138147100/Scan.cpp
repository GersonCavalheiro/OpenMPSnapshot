

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_iscan_(void *sendbuf, void *recvbuf,
MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op,
MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err);

void mpi_scan_(void *sendbuf, void *recvbuf,
MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op,
MPI_Fint *comm, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
MPI_Fint request;
mpi_iscan_(sendbuf, recvbuf, count, datatype, op, comm, &request, err);
if (*err == MPI_SUCCESS)
RequestManager<Fortran>::processRequest(request);
} else {
static mpi_scan_t *symbol = (mpi_scan_t *) Symbol::load(__func__);
(*symbol)(sendbuf, recvbuf, count, datatype, op, comm, err);
}
}

void tampi_iscan_internal_(void *sendbuf, void *recvbuf,
MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op,
MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err)
{
mpi_iscan_(sendbuf, recvbuf, count, datatype, op, comm, request, err);

if (Environment::isNonBlockingEnabled()) {
if (*err == MPI_SUCCESS) {
tampi_iwait_(request, MPI_F_STATUS_IGNORE, err);
}
}
}
} 

#pragma GCC visibility pop
