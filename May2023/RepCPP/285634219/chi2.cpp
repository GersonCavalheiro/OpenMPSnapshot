#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include "reference.h"

int main(int argc, char* argv[]) {

if (argc != 8) {
printf("Usage: %s <filename> <rows> <cols> <cases> <controls> "
"<threads> <repeat>\n", argv[0]);
return 1;
}

FILE *fp = fopen(argv[1], "r");
if (fp == NULL) {
printf("Cannot open the file %s\n", argv[1]);
return 1;
}

unsigned int rows = atoi(argv[2]);
unsigned int cols = atoi(argv[3]);
int ncases = atoi(argv[4]);
int ncontrols = atoi(argv[5]);
int nthreads = atoi(argv[6]);
int repeat = atoi(argv[7]);

printf("Individuals=%d SNPs=%d cases=%d controls=%d nthreads=%d\n",
rows,cols,ncases,ncontrols,nthreads);

size_t size = (size_t)rows * (size_t)cols;
printf("Size of the data = %lu\n",size);

unsigned char *dataT = (unsigned char*)malloc(size);
float* h_results = (float*) malloc(cols * sizeof(float)); 
float* cpu_results = (float*) malloc(cols * sizeof(float)); 

if(dataT == NULL || h_results == NULL || cpu_results == NULL) {
printf("ERROR: Memory for data not allocated.\n");
if (dataT) free(dataT);
if (h_results) free(h_results);
return 1;
}

#pragma omp target data map(to: dataT[0:size]) map(from: h_results[0:cols])
{
auto start = std::chrono::high_resolution_clock::now();

for (int i = 0; i < repeat; i++) {
#pragma omp target teams distribute parallel for simd thread_limit(nthreads)
for (int i = 0; i < cols; i++) {
unsigned char y;
int m, n;
unsigned int p = 0 ;
int cases[3];
int controls[3];
int tot_cases = 1;
int tot_controls= 1;
int total = 1;
float chisquare = 0.0f;
float exp[3];        
float Conexpected[3];        
float Cexpected[3];
float numerator1;
float numerator2;

cases[0]=1;cases[1]=1;cases[2]=1;
controls[0]=1;controls[1]=1;controls[2]=1;

for ( m = 0 ; m < ncases ; m++ ) {
y = dataT[m * cols + i];
if ( y == '0') { cases[0]++; }
else if ( y == '1') { cases[1]++; }
else if ( y == '2') { cases[2]++; }
}

for ( n = ncases ; n < ncases + ncontrols ; n++ ) {
y = dataT[n * cols + i];
if ( y == '0' ) { controls[0]++; }
else if ( y == '1') { controls[1]++; }
else if ( y == '2') { controls[2]++; }
}

tot_cases = cases[0]+cases[1]+cases[2];
tot_controls = controls[0]+controls[1]+controls[2];
total = tot_cases + tot_controls;

for( p = 0 ; p < 3; p++) {
exp[p] = (float)cases[p] + controls[p]; 
Cexpected[p] = tot_cases * exp[p] / total;
Conexpected[p] = tot_controls * exp[p] / total;
numerator1 = (float)cases[p] - Cexpected[p];
numerator2 = (float)controls[p] - Conexpected[p];
chisquare += numerator1 * numerator1 / Cexpected[p] +  numerator2 * numerator2 / Conexpected[p];
}
h_results[i] = chisquare;
}
}
auto end = std::chrono::high_resolution_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Average kernel execution time = %f (s)\n", time * 1e-9f / repeat);
}

auto start = std::chrono::high_resolution_clock::now();

cpu_kernel(rows,cols,ncases,ncontrols,dataT,cpu_results);

auto end = std::chrono::high_resolution_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("Host execution time = %f (s)\n", time * 1e-9f);

int error = 0;
for(unsigned int k = 0; k < cols; k++) {
if (fabs(cpu_results[k] - h_results[k]) > 1e-4) error++;
}
printf("%s\n", error ? "FAIL" : "PASS");

free(dataT);
free(h_results);
free(cpu_results);

return 0;
}
