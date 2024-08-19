
#ifndef SEIMS_MPI_PARALLEL_BASIC_H
#define SEIMS_MPI_PARALLEL_BASIC_H

#ifdef MSVC
#pragma warning(disable: 4819)
#endif 

#include "mpi.h"

#define WORK_TAG 0
#define MASTER_RANK 0
#define SLAVE0_RANK 1 
#define MAX_UPSTREAM 4
#define MSG_LEN 5
#define MCW MPI_COMM_WORLD

#endif 
