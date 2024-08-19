#ifndef OMEGA_H_MPI_H
#define OMEGA_H_MPI_H

#include <Omega_h_macros.h>

#ifdef OMEGA_H_USE_MPI

OMEGA_H_SYSTEM_HEADER


#ifdef __bgq__
#define MPICH2_CONST const
#define MPICH_SKIP_MPICXX
#endif

#ifdef __GNUC__
#pragma GCC diagnostic push
#if (__GNUC__ > 7) || (__GNUC__ == 7 && __GNUC_MINOR__ >= 3)
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#endif

#include <mpi.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif  

#endif  
