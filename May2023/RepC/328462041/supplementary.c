#include "supplementary.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>



void mergeSort_carry_one(double *arr, int *carry, int l, int r){
int m = -1;
if (l < r) {
m = l + (r - l) / 2;
#pragma omp task
mergeSort_carry_one(arr, carry, l, m);
mergeSort_carry_one(arr, carry, m + 1, r);
#pragma omp taskwait
merge_carry_one(arr, carry, l, m, r);
}
}

void merge_carry_one(double *arr, int *carry, int l, int m, int r){
int k;
int n1 = m - l + 1;
int n2 = r - m;

double *L = (double *)malloc(n1*sizeof(double));
if (L == NULL){
exit(1);
}
double *R = (double *)malloc(n2*sizeof(double));
if (R == NULL){
exit(1);
}
int *L_carry = (int *)malloc(n1*sizeof(int));
if (L_carry == NULL){
exit(1);
}
int *R_carry = (int *)malloc(n2*sizeof(int));
if (R_carry == NULL){
exit(1);
}

for (int i = 0; i < n1; i++){
L[i] = arr[l + i];
L_carry[i] = carry[l+i];
}
for (int j = 0; j < n2; j++){
R[j] = arr[m + 1 + j];
R_carry[j] = carry[m+1+j];
}

int i = 0;
int j = 0;
k = l;

while ((i < n1) && (j < n2)) {
if (L[i] <= R[j]) {
arr[k] = L[i];
carry[k] = L_carry[i];
i++;
}
else {
arr[k] = R[j];
carry[k] = R_carry[j];
j++;
}
k++;
}

while (i < n1) {
arr[k] = L[i];
carry[k] = L_carry[i];
i++;
k++;
}
while (j < n2) {
arr[k] = R[j];
carry[k] = R_carry[j];
j++;
k++;
}
free(L); free(R); free(L_carry); free(R_carry);
}


double time_spent(struct timespec start_time, struct timespec end_time){
struct timespec temp;
if ((end_time.tv_nsec - start_time.tv_nsec) < 0)
{
temp.tv_sec = end_time.tv_sec - start_time.tv_sec - 1;
temp.tv_nsec = 1000000000 + end_time.tv_nsec - start_time.tv_nsec;
}
else
{
temp.tv_sec = end_time.tv_sec - start_time.tv_sec;
temp.tv_nsec = end_time.tv_nsec - start_time.tv_nsec;
}
double returnval = (double)temp.tv_sec +(double)((double)temp.tv_nsec/(double)1000000000);

return returnval;
}


double quickselect(double *A, int left, int right, int k){
if (left == right){
return A[left];
}

int pivot = (left + right)/2;
pivot = partition_of_quick(A, left, right, pivot);

if (k == pivot){
return A[k];
} else if (k < pivot){
return quickselect(A, left, pivot - 1, k);
} else {
return quickselect(A, pivot + 1, right, k);
}
}

double partition_of_quick(double *a, int left, int right, int pivot){
double pivot_elem = a[pivot];
SWAP(&a[pivot], &a[right]);

int pIndex = left;
int i;

for (i = left; i < right; i++){
if (a[i] <= pivot_elem){
SWAP(&a[i], &a[pIndex]);
pIndex++;
}
}

SWAP(&a[pIndex], &a[right]);

return pIndex;
}

void SWAP(double *x, double *y) {
double temp = *x;
*x = *y;
*y = temp;
}


double *read_from_file(int *N, int *d){

FILE *f = fopen("FMA.txt","r");
if (f == NULL){
printf("File not found.\n");
exit(1);
}
*N = 106574;
*d = 518;

double *corpus_set = (double *)malloc(*N**d*sizeof(double));
if (corpus_set == NULL){
exit(1);
}
int counter;
for(int i=0; i<*N; i++){
for(int j=0; j<*d; j++){
if (j != (*d-1)){
counter = fscanf(f, "%lf\t",&corpus_set[i**d + j]);
}
else{
counter = fscanf(f, "%lf\n",&corpus_set[i**d + j]);
}
}
}
if (f != stdin){
fclose(f);
}
return corpus_set;
}
