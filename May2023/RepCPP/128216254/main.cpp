#include <iostream>
#include <mpi.h>
#include "ocular.h"
#include "lineSearch.h"
#include <math.h>
#include <iomanip>
#include <fstream>
#include "halfUtils.h"
#define MASTER 0

using namespace std;

int main(int argc, char *argv[])
{
int numProcs, rank;
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
ifstream inFile;
int tempTransfer[] = {0, 0, 0};
if (rank == MASTER) {
inFile.open("data/d2");
if(!inFile) {
cerr<<"Couldn't open file" << endl;
exit(0);
}
inFile >> tempTransfer[0] >> tempTransfer[1] >> tempTransfer[2];
}
MPI_Bcast(tempTransfer, 3, MPI_INT, MASTER, MPI_COMM_WORLD);
int numItems = tempTransfer[0];
int numUsers = tempTransfer[1];
int numRatings = tempTransfer[2];
int *csr_users = new int[numUsers + 1];
int *items = new int[numRatings];
int *csr_items = new int[numItems + 1];
int *users = new int[numRatings];
#pragma omp parallel 
{
#pragma omp for
for (int i = 0; i < numUsers + 1; i++) {
csr_users[i] = 0;
}
#pragma omp for
for (int i = 0; i < numItems + 1; i++) {
csr_items[i] = 0;
}
#pragma omp for
for (int i = 0; i < numRatings; i++) {
items[i] = users[i] = 0;
}
}
if (rank == MASTER) {
for (int i = 0; i < numRatings; i++) {
int item, user;
inFile >> item >> user;
user--;
users[i] = user;
csr_items[item]++;
}
for (int i = 1; i <= numItems; i++) {
csr_items[i] += csr_items[i - 1];
}

for (int i = 0; i < numRatings; i++) {
int user, item;
inFile >> item >> user;
item--;
items[i] = item;
csr_users[user]++;
}
for (int i = 1; i <= numUsers; i++) {
csr_users[i] += csr_users[i - 1];
}
}
MPI_Bcast(users, numRatings, MPI_INT, MASTER, MPI_COMM_WORLD);
MPI_Bcast(items, numRatings, MPI_INT, MASTER, MPI_COMM_WORLD);
MPI_Bcast(csr_users, numUsers + 1, MPI_INT, MASTER, MPI_COMM_WORLD);
MPI_Bcast(csr_items, numItems + 1, MPI_INT, MASTER, MPI_COMM_WORLD);

int process_items = numItems / numProcs + (numItems % numProcs > rank);
int process_users = numUsers / numProcs + (numUsers % numProcs > rank);

int *sendcounts_item = new int[numProcs];
int *displs_item = new int[numProcs];
int *sendcounts_user = new int[numProcs];
int *displs_user = new int[numProcs];

for (int i = 0; i < numProcs; i++) {
sendcounts_item[i] = numItems / numProcs + (numItems % numProcs > i);
if (i == 0)
displs_item[i] = 0;
else
displs_item[i] = displs_item[i - 1] + sendcounts_item[i - 1];
}

for (int i = 0; i < numProcs; i++) {
sendcounts_user[i] = numUsers / numProcs + (numUsers % numProcs > i);
if (i == 0)
displs_user[i] = 0;
else
displs_user[i] = displs_user[i - 1] + sendcounts_user[i - 1];
}

uint16_t **fi, **fu;
uint16_t *item_data = new uint16_t[numItems * CLUSTERS];
uint16_t *user_data = new uint16_t[numUsers * CLUSTERS];

fi = new uint16_t *[numItems];
fu = new uint16_t *[numUsers];

for (int i = 0; i < numItems; i++) {
fi[i] = &(item_data[i * CLUSTERS]);
}
for (int i = 0; i < numUsers; i++) {
fu[i] = &(user_data[i * CLUSTERS]);
}
ocular(numItems, numUsers, csr_items, users, csr_users, items, fi, fu, process_items, process_users, sendcounts_item, sendcounts_user, displs_item, displs_user, rank, numProcs);
if (rank == MASTER) {
float **fi_float, **fu_float;
float *item_data_f = new float[numItems * CLUSTERS];
float *user_data_f = new float[numUsers * CLUSTERS];

fi_float = new float *[numItems];
fu_float = new float *[numUsers];

for (int i = 0; i < numItems; i++) {
fi_float[i] = &(item_data_f[i * CLUSTERS]);
}
for (int i = 0; i < numUsers; i++) {
fu_float[i] = &(user_data_f[i * CLUSTERS]);
}
half2floatv(fu_float[0],fu[0],numUsers*CLUSTERS);
half2floatv(fi_float[0],fi[0],numItems*CLUSTERS);
int user_id, item_id, query_num;

}
MPI_Barrier(MPI_COMM_WORLD);
cout << "Done" <<endl;
MPI_Finalize();
return 0;
}
