

#include <mpi.h>

#include "include/TAMPI_Decl.h"

#include "Environment.hpp"
#include "Interface.hpp"
#include "RequestManager.hpp"
#include "Symbol.hpp"

using namespace tampi;

#pragma GCC visibility push(default)

extern "C" {
void mpi_igather_(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype,
void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err);

void mpi_gather_(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype,
void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
MPI_Fint *root, MPI_Fint *comm, MPI_Fint *err)
{
if (Environment::isBlockingEnabled()) {
MPI_Fint request;
mpi_igather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, &request, err);
if (*err == MPI_SUCCESS)
RequestManager<Fortran>::processRequest(request);
} else {
static mpi_gather_t *symbol = (mpi_gather_t *) Symbol::load(__func__);
(*symbol)(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, err);
}
}

void tampi_igather_internal_(void *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype,
void *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype,
MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *err)
{
mpi_igather_(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, request, err);

if (Environment::isNonBlockingEnabled()) {
if (*err == MPI_SUCCESS) {
tampi_iwait_(request, MPI_F_STATUS_IGNORE, err);
}
}
}
} 

#pragma GCC visibility pop
