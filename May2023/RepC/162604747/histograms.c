#if defined(__CYGWIN__) || defined(__MINGW64__)
#define _WIN32_WINNT 0x0602 
#include <Processtopologyapi.h>
#include <processthreadsapi.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <omp.h>
#include "mpfr.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <math.h>


void manage_thread_affinity()
{
#ifdef _WIN32_WINNT
int nbgroups = GetActiveProcessorGroupCount();
int *threads_per_groups = (int *) malloc(nbgroups*sizeof(int));
for (int i=0; i<nbgroups; i++)
{
threads_per_groups[i] = GetActiveProcessorCount(i);
}

int tid = omp_get_thread_num(); 
HANDLE thandle = GetCurrentThread();
_Bool result;

int set_group = tid%nbgroups; 
int nbthreads = threads_per_groups[set_group]; 
GROUP_AFFINITY group = {((uint64_t)1<<nbthreads)-1, set_group}; 

result = SetThreadGroupAffinity(thandle, &group, NULL); 
if(!result) fprintf(stderr, "Failed setting output for tid=%i\n", tid);
free(threads_per_groups);
#else
#endif
}



void swap_histogram(uint64_t *hist, const int b)
{
const int halfsize = 1<<(b-1);
uint64_t *tmp = calloc(halfsize, sizeof(uint64_t));
int i=0;
for (; i<halfsize; i++){  
tmp[i] = hist[i];
hist[i] = hist[i+halfsize];
}
for (; i<2*halfsize; i++){
hist[i] = tmp[i-halfsize];
}
free(tmp);
}


void histogram8_unsigned(uint8_t *data, uint64_t size, uint64_t *hist)
{   
uint64_t *data_64 = (uint64_t *) data;
#pragma omp parallel
{
manage_thread_affinity(); 
uint64_t tmp=0;
#pragma omp for reduction(+:hist[:1<<8])
for (uint64_t i=0; i<size/8; i++){
tmp = data_64[i];
hist[tmp >>  0 & 0xFF]++;
hist[tmp >>  8 & 0xFF]++;
hist[tmp >> 16 & 0xFF]++;
hist[tmp >> 24 & 0xFF]++;
hist[tmp >> 32 & 0xFF]++;
hist[tmp >> 40 & 0xFF]++;
hist[tmp >> 48 & 0xFF]++;
hist[tmp >> 56 & 0xFF]++;
}
}
for (uint64_t i=size-(size%8); i<size; i++){
hist[ data[i] ]++; 
}
}


void histogram8_signed(int8_t *data, uint64_t size, uint64_t *hist)
{
uint8_t *data_unsigned = (uint8_t *) data;
histogram8_unsigned(data_unsigned, size, hist);
swap_histogram(hist, 8); 
}


void histogram16_unsigned(uint16_t *data, uint64_t size, uint64_t *hist, const int b)
{   
const int32_t tail = 16-b;
uint64_t *data_64 = (uint64_t *) data;
#pragma omp parallel
{
manage_thread_affinity(); 
uint64_t tmp=0;
#pragma omp for reduction(+:hist[:1<<b])
for (uint64_t i=0; i<size/4; i++){
tmp = data_64[i]; 
hist[ (tmp >>   0 & 0xFFFF) >> tail ]++;
hist[ (tmp >>  16 & 0xFFFF) >> tail ]++;
hist[ (tmp >>  32 & 0xFFFF) >> tail ]++;
hist[ (tmp >>  48 & 0xFFFF) >> tail ]++;
}
}
for (uint64_t i=size-(size%4); i<size; i++){
hist[ data[i] >> tail ]++; 
}
}


void histogram16_signed(int16_t *data, uint64_t size, uint64_t *hist, const int b)
{
uint16_t *data_unsigned = (uint16_t *) data;
histogram16_unsigned(data_unsigned, size, hist, b);
swap_histogram(hist, b);
}


void swap_histogram2d(uint64_t *hist, const int b)
{
uint64_t rsize = 1<<b;     
swap_histogram(hist, 2*b); 
#pragma omp parallel
{
manage_thread_affinity();
#pragma omp for
for (uint64_t i=0; i<rsize; i++){ 
swap_histogram(hist+(i*rsize), b); 
}
}
}


void histogram2d8_unsigned(uint8_t *data1, uint8_t *data2, uint64_t size, uint64_t *hist)
{
uint64_t *data1_64 = (uint64_t *) data1;
uint64_t *data2_64 = (uint64_t *) data2;
#pragma omp parallel
{
manage_thread_affinity(); 
uint64_t tmp1=0;
uint64_t tmp2=0;
#pragma omp for reduction(+:hist[:1<<(8*2)])
for (uint64_t i=0; i<size/8; i++){
tmp1 = data1_64[i]; 
tmp2 = data2_64[i]; 
hist[ (tmp1 <<  8 & 0xFF00) + (tmp2 >>  0 & 0xFF) ]++;
hist[ (tmp1 >>  0 & 0xFF00) + (tmp2 >>  8 & 0xFF) ]++;
hist[ (tmp1 >>  8 & 0xFF00) + (tmp2 >> 16 & 0xFF) ]++;
hist[ (tmp1 >> 16 & 0xFF00) + (tmp2 >> 24 & 0xFF) ]++;
hist[ (tmp1 >> 24 & 0xFF00) + (tmp2 >> 32 & 0xFF) ]++;
hist[ (tmp1 >> 32 & 0xFF00) + (tmp2 >> 40 & 0xFF) ]++;
hist[ (tmp1 >> 40 & 0xFF00) + (tmp2 >> 48 & 0xFF) ]++;
hist[ (tmp1 >> 48 & 0xFF00) + (tmp2 >> 56 & 0xFF) ]++;
}
}
for (uint64_t i=size-(size%8); i<size; i++){
hist[ (data1[i]<<8) + data2[i] ]++;
}
}


void histogram2d8_signed(int8_t *data1, int8_t *data2, uint64_t size, uint64_t *hist)
{
uint8_t *data1_unsigned = (uint8_t *) data1;
uint8_t *data2_unsigned = (uint8_t *) data2;
histogram2d8_unsigned(data1_unsigned, data2_unsigned, size, hist);
swap_histogram2d(hist, 8); 
}


void reduce(uint64_t** arrs, uint64_t bins, uint64_t begin, uint64_t end)
{
assert(begin < end);
if (end - begin == 1) {
return;
}
uint64_t pivot = (begin + end) / 2;

reduce(arrs, bins, begin, pivot);
reduce(arrs, bins, pivot, end);

#pragma omp parallel 
{
manage_thread_affinity();
#pragma omp for
for (uint64_t i = 0; i < bins; i++)
arrs[begin][i] += arrs[pivot][i];
}
}

void histogram2d16_unsigned(uint16_t *data1, uint16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic)
{
const int32_t tail0 = 16-b;
const int32_t tail1 = tail0+16;
const int32_t tail2 = tail1+16;
const int32_t tail3 = tail2+16;
const int32_t mask = (1<<b)-1; 

uint64_t *data1_64 = (uint64_t *) data1;
uint64_t *data2_64 = (uint64_t *) data2;
if (atomic==1){
#pragma omp parallel
{
manage_thread_affinity(); 

uint64_t tmp1=0;
uint64_t tmp2=0;
#pragma omp for 
for (uint64_t i=0; i<size/4; i++){
tmp1 = data1_64[i]; 
tmp2 = data2_64[i]; 
#pragma omp atomic update
hist[ ((tmp1 >> tail0 & mask) << b) + (tmp2 >> tail0 & mask) ]++;   
#pragma omp atomic update
hist[ ((tmp1 >> tail1 & mask) << b) + (tmp2 >> tail1 & mask) ]++;
#pragma omp atomic update
hist[ ((tmp1 >> tail2 & mask) << b) + (tmp2 >> tail2 & mask) ]++;
#pragma omp atomic update
hist[ ((tmp1 >> tail3 & mask) << b) + (tmp2 >> tail3 & mask) ]++;
}
}
}
else{
uint64_t **hs;
int n;
#pragma omp parallel
{
manage_thread_affinity(); 
n = omp_get_num_threads(); 

#pragma omp single 
hs = (uint64_t **) malloc(n * sizeof(uint64_t));
uint64_t *h = (uint64_t *) calloc(1<<(b*2), sizeof(uint64_t)); 
hs[omp_get_thread_num()] = h;

uint64_t tmp1=0;
uint64_t tmp2=0;
#pragma omp for nowait
for (uint64_t i=0; i<size/4; i++){
tmp1 = data1_64[i]; 
tmp2 = data2_64[i]; 
h[ ((tmp1 >> tail0 & mask) << b) + (tmp2 >> tail0 & mask) ]++;   
h[ ((tmp1 >> tail1 & mask) << b) + (tmp2 >> tail1 & mask) ]++;
h[ ((tmp1 >> tail2 & mask) << b) + (tmp2 >> tail2 & mask) ]++;
h[ ((tmp1 >> tail3 & mask) << b) + (tmp2 >> tail3 & mask) ]++;
}
}
reduce(hs, 1<<(b*2), 0, n); 
#pragma omp parallel
{
manage_thread_affinity();
#pragma omp for
for (uint64_t i=0; i<1<<(b*2); i++){
hist[i]+=hs[0][i];
}
}
for (int i=0; i<n; i++){
free(hs[i]);
}
free(hs);
}

for (uint64_t i=size-(size%4); i<size; i++){
hist[ ((data1[i]>>tail0)<<b) + (data2[i]>>tail0) ]++;
}
}


void histogram2d16_signed(int16_t *data1, int16_t *data2, uint64_t size, uint64_t *hist, const uint32_t b, const int atomic)
{
uint16_t *data1_unsigned = (uint16_t *) data1;
uint16_t *data2_unsigned = (uint16_t *) data2;
histogram2d16_unsigned(data1_unsigned, data2_unsigned, size, hist, b, atomic);
swap_histogram2d(hist, b);
}

int64_t nCk(int n, int k)
{
if (k==0){
return 1;
}
return (n*nCk(n-1, k-1))/k; 
}

double moment(uint64_t *hist, const int b, const int k, const int centered)
{
const int size = 1<<b;
long double bshift=0;
long double val = 0;
uint64_t n=0;

if (centered){
bshift = moment(hist, b, 1, 0);
}
#pragma omp parallel
{
manage_thread_affinity(); 
if (centered){
#pragma omp for reduction(+:val), reduction(+:n)
for (int i=0; i<size; i++){
val += (long double)hist[i] * powl((long double)i - (long double)bshift, k);
n += hist[i];
}
}
else{
#pragma omp for reduction(+:val), reduction(+:n)
for (int i=0; i<size; i++){
val += (long double)hist[i] * powl((long double)i, k);
n += hist[i];
}
}
}
return (double)(val/(long double)n);
}

double cumulant(uint64_t *hist, const int b, const int k){
double ret = moment(hist, b, k, 0);
for (int i=1; i<k; i++){
ret -= (double)nCk(k-1, i-1)*cumulant(hist, b, i)*moment(hist, b, k-i, 0);
}
return ret;
}
