#include <iostream>
#include "mpi.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>
using namespace std;

int main(int argc, char **argv)
{
int *arr, rank, root = 0, nprocs, namelen;
char processorName[10];

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
MPI_Get_processor_name(processorName, &namelen);

if (nprocs != 1) {  
int size = 20;  
int size_of_arrays = size/(nprocs - 1);     
int number_to_search = 31;   

if (rank == root) {     
cout << "Name: Sameet Asadullah\nRoll Number: 18i-0479\n\n";

cout << processorName << ": The number to search is " << number_to_search << "\n";
cout << "Process 0 has input data: "; 
arr = new int[size];
int temp = 0;
for (int i = 1; i < size * 2; i += 2) {    
arr[temp] = i;
cout << arr[temp] << " ";
temp++;
}
cout << "\n";

for (int i = 1, j = 0; i < nprocs; ++i) {   
int *arr_to_send = new int[size_of_arrays];
for (int k = 0; k < size_of_arrays; ++k, ++j) {
arr_to_send[k] = arr[j];
}
MPI_Send(arr_to_send, size_of_arrays, MPI_INT, i, 1234, MPI_COMM_WORLD);
delete[] arr_to_send;
}

if (nprocs != 2) {  
MPI_Status status;
char data[27];
MPI_Recv(data, 27, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

if (strcmp(data, "I have found the number :-)") == 0) {
sleep(1);
cout << processorName << ": Process " << status.MPI_SOURCE << " has found the number!\n";
cout << processorName << ": Informing all processes to abort!\n";
for (int i = 1; i < nprocs; ++i) {  
char abort_message[6] = "ABORT";
MPI_Send(abort_message, 6, MPI_CHAR, i, 1236, MPI_COMM_WORLD);
}
}
}
} else {    
arr = new int[size_of_arrays];
MPI_Recv(arr, size_of_arrays, MPI_INT, 0, 1234, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

cout << "Process " << rank << " has local data: ";
for (int i = 0; i < size_of_arrays; ++i) {
cout << arr[i] << " ";
}
cout << "\n";
sleep(2);

char abort_message[6];
#pragma omp parallel num_threads(2) 
{
if (omp_get_thread_num() == 0) {    
for (int i = 0; i < size_of_arrays; ++i) {
if (strcmp(abort_message, "ABORT") == 0) {  
cout << "Process " << rank << ": Aborting search!\n";
break;
}

if (arr[i] == number_to_search) {   
char message[] = "I have found the number :-)";
MPI_Send(message, 27, MPI_CHAR, 0, 1235, MPI_COMM_WORLD);
cout << "Process " << rank << ": " << message << "\n";
break;
}

if (rank != 4) {
sleep(2);
}
}
} else {    
if (nprocs != 2) {
MPI_Recv(abort_message, 6, MPI_CHAR, 0, 1236, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
}
}
delete[] arr;
}
} else {
cout << "No. of processes must be more than 1." << endl;
}
MPI_Finalize();
}