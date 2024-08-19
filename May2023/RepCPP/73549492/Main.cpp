#include <iostream>
#include <omp.h>
#include <mpi.h>

#include "Master.hpp"
#include "Slave.hpp"


int main(int argc, char** argv) {
MPI_Init(&argc, &argv);

int machineCount;
int rank;
MPI_Comm_size(MPI_COMM_WORLD, &machineCount);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
std::cout << "Number of machines in the cluster: " << machineCount << std::endl;
std::cout << "This machine's rank: " << rank << std::endl;

#pragma omp parallel
#pragma omp master
{
std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
}

if (rank == 0) {
std::cout << "\nSTARTING MASTER\n\n";
Master::Create()->Run(machineCount);
}
else {
std::cout << "\nSTARTING SLAVE\n\n";
Slave::Create()->Run();
}

MPI_Finalize();

return 0;
}