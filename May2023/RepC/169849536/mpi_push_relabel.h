#pragma once
#include "mpi.h"
int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow);
namespace utils {
int idx(int x, int y, int n);
}