#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <vector>
std::vector<int> test(int *array, int threads) {
std::vector<int> out;
for (int i = array[0]; i < array[1]; i++) {
int length = sqrt(i);
bool flag = false;
#pragma omp parallel num_threads(threads) shared(i, length, threads) default(none)
{
#pragma omp for
for (int j = 2; j < length + 1; j++) {
if (!(i % j)) {
flag = true;
}
}
}
if (!flag) {
out.push_back(i);
}
}
return out;
}
int main(int argc, char **argv) {
int ret = -1;
int size = -1;
int rank = -1;
const int threads = 2;
const int lower_bound = 2;
const int upper_bound = 5000000;
int *array = 0;
int *dArray = 0;
int *sArray = 0;
int array_size = 0;
int array_step = 1;
double start_time, end_time;
MPI_Status *status;
ret = MPI_Init(&argc, &argv);
if (!rank) {
printf("MPI_Init return value: %d\n", ret);
exit(0);
}
MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
if (!rank) {
start_time = MPI_Wtime();
array = new int[2];
array_step = (upper_bound - lower_bound) / size;
for (int i = 1; i < size; i++) {
array[0] = lower_bound + array_step * i;
array[1] = lower_bound + array_step * (i + 1);
MPI_Send(array, 2, MPI_INT, i, 1, MPI_COMM_WORLD);
}
array[0] = lower_bound;
array[1] = lower_bound + array_step;
dArray = new int[size];
std::vector<int> subArray = test(array, threads);
std::vector<int> rArray[size];
rArray[0] = subArray;
dArray[0] = subArray.size();
for (int i = 1; i < size; i++) {
MPI_Recv(dArray + i, 1, MPI_INT, i, 2, MPI_COMM_WORLD, status);
sArray = new int[dArray[i]];
MPI_Recv(sArray, dArray[i], MPI_INT, i, 3, MPI_COMM_WORLD, status);
std::vector<int> vector;
for (int j = 0; j < dArray[i]; j++) {
vector.push_back(sArray[j]);
}
rArray[i] = vector;
}
}
else {
array = new int[2];
MPI_Recv(array, 2, MPI_INT, 0, 1, MPI_COMM_WORLD, status);
start_time = MPI_Wtime();
std::vector<int> subArray = test(array, threads);
array_size = subArray.size();
MPI_Send(&array_size, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
MPI_Send(&subArray[0], array_size, MPI_INT, 0, 3, MPI_COMM_WORLD);
}
end_time = MPI_Wtime();
printf("==================================\nrank: %d || time elapsed: %0.7lf\n==================================\n", rank, end_time - start_time);
ret = MPI_Finalize();
return 0;
}