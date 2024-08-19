#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char** argv){

int myrank, nprocs;
int tag = 100;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

if (myrank == 0) {

int size = (nprocs - 1) * 3;
int* list = new int[size];

for (int i = 0; i < size; i++)
list[i] = rand() % 1000;

cout << "Array Elements: ";
for (int i = 0; i < size; i++)
cout << list[i] << " ";
cout << endl;

int key = rand() % size;
key = list[key];

cout << "Master: The number to search is " << key << endl;


int num_elements = size / (nprocs-1);
for(int i = 1; i < nprocs; i++) {
MPI_Send(&num_elements, 1, MPI_INT, i, tag, MPI_COMM_WORLD);

int* temp = new int[3];
for(int c = 0; c < 3; c++)
temp[c] = list[(3*(i-1))+c];

MPI_Send(temp, 3, MPI_INT, i, tag, MPI_COMM_WORLD);
MPI_Send(&key, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
}

MPI_Status status;
int dummy;
MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

cout << "Master: Process " << status.MPI_SOURCE << " has found the number!" << endl;
cout << "Master: Informing all processes to abort!" << endl;
for(int i = 1; i < nprocs; i++) {
if (i != status.MPI_TAG)
MPI_Send(&dummy, 1, MPI_INT, i, tag, MPI_COMM_WORLD);
}
}
else{

int key, num_elements;

MPI_Recv(&num_elements, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

int* list =  new int[num_elements];

MPI_Recv(list, 3, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
MPI_Recv(&key, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

bool isFound = false;
int tid;
#pragma omp parallel num_threads(2)
{
tid = omp_get_thread_num();
if (tid == 0) {

for (int i = 0; i < num_elements; i++) {
if (isFound) {
cout << "Process " << myrank << ": Aborting Search!" << endl;
break;
}

if (list[i] == key) {
cout << "Process " << myrank << ": I have found the number!" << endl;
int dummy = 1;
MPI_Send(&dummy, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
break;
}
}
}
else {
int dummy;
MPI_Recv(&dummy, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
isFound = true;
}
}
}

MPI_Finalize();
return 0;
}
