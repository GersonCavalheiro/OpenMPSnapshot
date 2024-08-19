#pragma once

#include <mpi.h>

namespace quids::mpi::utils {

void make_equal_pairs(size_t *size_begin, size_t *size_end, int *pair_id) {
size_t size = std::distance(size_begin, size_end);

std::vector<int> node_ids(size, 0);
std::iota(node_ids.begin(), node_ids.begin() + size, 0);


__gnu_parallel::sort(node_ids.begin(), node_ids.begin() + size,
[&](int const node_id1, int const node_id2) {
return size_begin[node_id1] > size_begin[node_id2];
});

#pragma omp parallel for
for (int i = 0; i < size; ++i)
pair_id[node_ids[i]] = node_ids[size - i - 1];
} 


template<typename T>
MPI_Datatype get_mpi_datatype(T x) { return NULL; }
MPI_Datatype get_mpi_datatype(float x) { return MPI_FLOAT; }
MPI_Datatype get_mpi_datatype(double x) { return MPI_DOUBLE; }
MPI_Datatype get_mpi_datatype(long double x) { return MPI_LONG_DOUBLE; }
MPI_Datatype get_mpi_datatype(std::complex<float> x) { return MPI_COMPLEX; }
MPI_Datatype get_mpi_datatype(std::complex<double> x) { return MPI_DOUBLE_COMPLEX; }
MPI_Datatype get_mpi_datatype(std::complex<long double> x) { return MPI_C_LONG_DOUBLE_COMPLEX; }
MPI_Datatype get_mpi_datatype(bool x) { return MPI_CHAR; }
MPI_Datatype get_mpi_datatype(char x) { return MPI_CHAR; }
MPI_Datatype get_mpi_datatype(signed char x) { return MPI_SIGNED_CHAR; }
MPI_Datatype get_mpi_datatype(unsigned char x) { return MPI_UNSIGNED_CHAR; }
MPI_Datatype get_mpi_datatype(short x) { return MPI_SHORT; }
MPI_Datatype get_mpi_datatype(unsigned short x) { return MPI_UNSIGNED_SHORT; }
MPI_Datatype get_mpi_datatype(int x) { return MPI_INT; }
MPI_Datatype get_mpi_datatype(unsigned int x) { return MPI_UNSIGNED; }
MPI_Datatype get_mpi_datatype(long x) { return MPI_LONG; }
MPI_Datatype get_mpi_datatype(unsigned long x) { return MPI_UNSIGNED_LONG; }
}