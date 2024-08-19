#pragma once

#include "monolish/common/monolish_common.hpp"
#include <climits>
#include <vector>

#if defined MONOLISH_USE_MPI
#include <mpi.h>
#else
#include "monolish/mpi/mpi_dummy.hpp"
#endif

#if SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#endif

#include "monolish/mpi/monolish_mpi_core.hpp"
