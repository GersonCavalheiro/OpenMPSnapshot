

#ifndef TAMPI_DECL_H
#define TAMPI_DECL_H

#include <mpi.h>

#pragma GCC visibility push(default)

#define MPI_TASK_MULTIPLE (MPI_THREAD_MULTIPLE + 1)

#ifdef __cplusplus
extern "C" {
#endif

int TAMPI_Blocking_enabled(int *flag);
int TAMPI_Nonblocking_enabled(int *flag);
int TAMPI_Iwait(MPI_Request *request, MPI_Status *status);
int TAMPI_Iwaitall(int count, MPI_Request array_of_requests[], MPI_Status array_of_statuses[]);

void tampi_blocking_enabled_(MPI_Fint *flag, MPI_Fint *err);
void tampi_nonblocking_enabled_(MPI_Fint *flag, MPI_Fint *err);
void tampi_iwait_(MPI_Fint *request, MPI_Fint *status, MPI_Fint *err);
void tampi_iwaitall_(MPI_Fint *count, MPI_Fint array_of_requests[], MPI_Fint *array_of_statuses, MPI_Fint *err);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif 
