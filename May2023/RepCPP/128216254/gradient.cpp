

#include "omp.h"
#include "lineSearch.h"
#include "gradient.h"
#include <cmath>
#include <iostream>
#include "mpi.h"
#include <ctime>
#include "halfUtils.h"

using namespace std;
void gradient(float **items, float **users, int start_index, int numItems, int numUser, int totalUser, int *csr_items, int *csr_users, float *user_sum, float **g, MPI_Request &mpi_req, uint16_t **short_users)
{
#pragma omp parallel for
for (int i = 0; i < numItems; i++) {
int item = start_index + i;
for (int j = 0; j < CLUSTERS; j++) {
g[item][j] =2 * LAMBDA * items[item][j];
}
}
MPI_Wait(&mpi_req, MPI_STATUS_IGNORE);
half2floatv(users[0], short_users[0], totalUser*CLUSTERS);
for (int i = 0; i < totalUser; i++) { 
for (int j = 0; j < CLUSTERS; j++)	
user_sum[j] += users[i][j];
}
#pragma omp parallel for
for (int i = 0; i < numItems; i++) {
int item = start_index + i;
for (int j = 0; j < CLUSTERS; j++){
g[item][j] += user_sum[j];
}
for (int user_idx = csr_items[item]; user_idx < csr_items[item + 1]; user_idx++) {
int user = csr_users[user_idx];
float x = innerProduct(items[item], users[user], CLUSTERS);
float factor = 1.0 / (1 -  exp(-x));
if(x==0)
factor=-factor;
for (int j = 0; j < CLUSTERS; j++) {
if(users[user][j]==0 or factor==0)
g[item][j]-= 0;
else
g[item][j] -= users[user][j] * factor;
}
}
}
}
