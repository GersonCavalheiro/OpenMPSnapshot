#include "../include/openMpHamming.h"
#include <stdio.h>
#include <omp.h>
double openMpHamm_taskA(structs *src, unsigned long long serialHammingSum) {
unsigned long long sum = 0;
int **hammingValues = init2dArray(src->Alen, src->Blen);
if(THREADS!=BEST){
omp_set_num_threads(THREADS);
}
printf("OpenMP task A.....");
double begin = gettime();
#pragma omp parallel
{
int i,j,k,t;
unsigned long long psum=0;
int NUM_THREADS=omp_get_num_threads();
int THREAD_ID=omp_get_thread_num();
for(t=0;t<NUM_THREADS;t++){
int start=(t+THREAD_ID)%NUM_THREADS;
for(i=start;i<src->Alen;i=i+NUM_THREADS){
for (j = 0; j < src->Blen; j++) {
#pragma omp for nowait
for (k = 0; k < src->Strlen; k++) {
if (src->A[i][k] != src->B[j][k]) {
psum++;
#pragma omp atomic
hammingValues[i][j]++;
}
}
}
}
}
#pragma omp atomic
sum += psum;
}
double end = gettime();
double calcTime = end - begin;
if (serialHammingSum != sum) {
printf(ANSI_RED "Error!"ANSI_RESET"\n");
return (double) (-1);
}
printf(ANSI_GREEN"finished"ANSI_RESET"\t ");
printf("Hamming time:%f sec ", calcTime);
if(DEBUG) { printf("| Sum Value:%lld", calcSumOfArray(src->Alen, src->Blen, hammingValues)); }
printf("\n");
return calcTime;
}
double openMpHamm_taskB(structs *src, unsigned long long serialHammingSum) {
unsigned long long sum = 0;
int **hammingValues = init2dArray(src->Alen, src->Blen);
if(THREADS!=BEST){
omp_set_num_threads(THREADS);
}
printf("OpenMP task B.....");
double begin = gettime();
#pragma omp parallel
{
int i, j, k;
unsigned long long psum = 0;
for(k=0;k<src->Strlen;k++){
#pragma omp for collapse(2) nowait
for(i=0;i<src->Alen;i++){
for(j=0;j<src->Blen;j++){
if (src->A[i][k] != src->B[j][k]) {
hammingValues[i][j]++;
psum++;
}
}
}
}
#pragma omp atomic
sum += psum;
}
double end = gettime();
double calcTime = (double) (end - begin);
if (serialHammingSum != sum) {
printf(ANSI_RED "Error!"ANSI_RESET"\n");
return (double) (-1);
}
printf(ANSI_GREEN"finished"ANSI_RESET"\t ");
printf("Hamming time:%f sec ", calcTime);
if(DEBUG) { printf("| Sum Value:%lld", calcSumOfArray(src->Alen, src->Blen, hammingValues));}
printf("\n");
return calcTime;
}
double openMpHamm_taskC(structs *src, unsigned long long serialHammingSum) {
unsigned long long sum = 0;
int **hammingValues = init2dArray(src->Alen, src->Blen);
if(THREADS!=BEST){
omp_set_num_threads(THREADS);
}
printf("OpenMP task C.....");
double begin = gettime();
#pragma omp parallel
{
int i, j, k;
unsigned long long psum = 0;
for (i = 0; i < src->Alen; i++) {
for (k = 0; k < src->Strlen; k++) {
#pragma omp for nowait
for (j = 0; j < src->Blen; j++) {
if (src->A[i][k] != src->B[j][k]) {
hammingValues[i][j]++;
psum++;
}
}
}
}
#pragma omp atomic
sum += psum;
}
double end = gettime();
double calcTime = (double) (end - begin);
if (serialHammingSum != sum) {
printf(ANSI_RED "Error!"ANSI_RESET"\n");
return (double) (-1);
}
printf(ANSI_GREEN"finished"ANSI_RESET"\t ");
printf("Hamming time:%f sec ", calcTime);
if(DEBUG) {  printf("| Sum Value:%lld", calcSumOfArray(src->Alen, src->Blen, hammingValues));}
printf("\n");
dealhamm(src->Alen,src->Blen,hammingValues);
return calcTime;
}