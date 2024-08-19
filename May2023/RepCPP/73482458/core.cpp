
#include <mpi.h>

#include "heat.hpp"

void exchange(Field& field, const ParallelData parallel)
{

double* sbuf = field.temperature.data(1, 0);
double* rbuf  = field.temperature.data(field.nx + 1, 0);
MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE,
parallel.nup, 11,
rbuf, field.ny + 2, MPI_DOUBLE, 
parallel.ndown, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

sbuf = field.temperature.data(field.nx, 0);
rbuf = field.temperature.data();
MPI_Sendrecv(sbuf, field.ny + 2, MPI_DOUBLE, 
parallel.ndown, 12,
rbuf, field.ny + 2, MPI_DOUBLE,
parallel.nup, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

#pragma omp for
for (int i = 1; i < curr.nx + 1; i++) {
for (int j = 1; j < curr.ny + 1; j++) {
curr(i, j) = prev(i, j) + a * dt * (
( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
);
}
}

}
