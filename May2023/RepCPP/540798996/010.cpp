
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <conio.h>
using namespace std;

int main(){

int count_treads = 50;

int n = 100000; 

int *b = (int*) calloc(n+1, sizeof(int));
int** a;
a = (int**)calloc(n+1, sizeof(int*));
for (int i = 0; i < n; i++) {
a[i] = (int*)calloc(n+1, sizeof(int));
}


int max_sec, max_par;

double time_spent = 0.00000000;
double time_spent_par = 0.00000000;

for(int i = 0; i < n; ++i) {
for(int j = 0; j < n; ++j) {
a[i][j] = rand();
}
}


clock_t begin_par =  clock();
#pragma omp parallel for num_threads(count_treads)
for(int i = 0; i < n; ++i) {
b[i] = a[i][0];
#pragma omp parallel for num_threads(count_treads)
for(int j = 0; j < n; ++j) {
if (a[i][j] < b[i]) {
b[i] = a[i][j];
}
}
}
max_par = b[0];
#pragma omp parallel for num_threads(count_treads)
for (int i = 0; i < n; ++i) {
if (b[i] > max_par) {
max_par = b[i];
}
}
clock_t end_par =  clock();
time_spent_par += (double)(end_par - begin_par) / (CLOCKS_PER_SEC);
printf("\n Max Parallel num %i", max_par);
printf("\nParallel work time is %.10f seconds", time_spent_par);

free(a);
free(b);
return 0;
}