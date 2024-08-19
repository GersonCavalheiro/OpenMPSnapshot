#include <omp.h>
#include "lineSearch.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <iostream>
using namespace std;


float innerProduct(float *A, float *B, int size) {
float val = 0;
for (int i = 0; i < size; i++) {
val = val + A[i] * B[i];
}
return val;
}

void likelihood(float *Q, bool *selected, float *user_sum, float **items, float **users, int numItems,
int *item_sparse_csr_r, int *user_sparse_csr_c, int start_index, int totalItems, bool type) {
#pragma omp parallel for
for (int i = 0; i < numItems; i++) {
if (selected[i]) {
if (type) 
Q[i] = innerProduct(items[start_index + i], user_sum, CLUSTERS) +
LAMBDA * innerProduct(items[start_index + i], items[start_index + i], CLUSTERS);
else
Q[i] = innerProduct(items[i], user_sum, CLUSTERS) + LAMBDA * innerProduct(items[i], items[i], CLUSTERS);

int start = item_sparse_csr_r[start_index + i];
int end = item_sparse_csr_r[start_index + i + 1];
for (int j = start; j < end; j++) {
int uid = user_sparse_csr_c[j];
float x;
if (type)
x = innerProduct(items[start_index + i], users[uid], CLUSTERS);
else
x = innerProduct(items[i], users[uid], CLUSTERS);
float y = Q[i];
Q[i] = Q[i] - x - log(1 - exp(-x)); 
}
}
}
}

void linesearch(float **items, float *user_sum, float **users, float **gradient, int numItems, int start_index, int totalItems,
int *item_sparse_csr_r, int *user_sparse_csr_c) {
float **newItems, **tempItems;
tempItems = new float *[numItems];
newItems = new float *[numItems];

for (int i = 0; i < numItems; i++) {
newItems[i] = new float[CLUSTERS];
tempItems[i] = new float[CLUSTERS];
}
bool *active = new bool[numItems];
memset(active, true, numItems * sizeof(bool));
float *Q = new float[numItems];
float *Q2 = new float[numItems];
likelihood(Q, active, user_sum, items, users, numItems, item_sparse_csr_r, user_sparse_csr_c, start_index, totalItems,
true);
double alpha = 1;
bool flag = true;
int removed = 0;
while (flag) {
#pragma omp parallel for
for (int i = 0; i < numItems; i++) {
if (active[i])
for (int j = 0; j < CLUSTERS; j++) {
float newVal = items[start_index + i][j] - alpha * gradient[start_index + i][j];
newItems[i][j] = max(newVal, 0.0f);
assert(!isnan(newItems[i][j]));
}
}
likelihood(Q2, active, user_sum, newItems, users, numItems, item_sparse_csr_r, user_sparse_csr_c, start_index,
totalItems, false);
int reduce_remove=0;
#pragma omp parallel for reduction( + : reduce_remove)
for (int i = 0; i < numItems; i++) {
if (active[i]) {
for (int j = 0; j < CLUSTERS; j++)
tempItems[i][j] = newItems[i][j] - items[start_index + i][j];

if (Q2[i] - Q[i] <= SIGMA * innerProduct(gradient[start_index + i], tempItems[i], CLUSTERS)) {
active[i] = false;
reduce_remove++;
}

}
}
alpha = alpha * BETA;
removed+=reduce_remove;
if (removed == numItems)
flag = false;
}
delete[] active;
delete[] Q;
delete[] Q2;
for (int i = 0; i < numItems; i++) {
for (int j = 0; j < CLUSTERS; j++) {
items[start_index + i][j] = newItems[i][j];
}
}

for (int i = 0; i < numItems; i++) {
delete[] newItems[i];
delete[] tempItems[i];
}
delete[] newItems;
delete[] tempItems;
}
