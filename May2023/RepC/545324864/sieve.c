#include <omp.h>
#include <stdio.h>
#include <limits.h>
#define DEBUG 0
#ifndef THREADS
#define THREADS 1
#endif
#ifndef DIM
#define DIM 10
#endif
#define MAX_ELEMENT_RANGE INT_MAX
FILE *fp;
int a_seq[DIM];
int a_par[DIM];
void initialize();
int isqrt(int value);
void sieve_sequential();
void sieve_parallel();
int test_equality();
int main(void){
omp_set_num_threads(THREADS);
printf("! Utilizing %d threads\n", THREADS);
initialize();
sieve_sequential();
sieve_parallel();
int res = test_equality();
printf("! Sieve success: %d\n", res);
if(res != 0){
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Sieve of Erastothenes error: Parallel and Sequential results differ.");
fclose(fp);
}
return 0;
}
void initialize(){
if(DEBUG) printf("\tInitializing arrays\n");
for(int i = 0; i < DIM; i++){
if(DEBUG) printf("\tOverwriting arrays with 0 at %d\n", i);
a_seq[i] = 0;
a_par[i] = 0;
}
}
int isqrt(int value){
int res = 0;
while ((res + 1) * (res + 1) <= value)
{
res = res + 1;
if(DEBUG) printf("\tRunning sqrt: %d\n", res);
}
if(DEBUG) printf("\tSQRT(%d) = %d\n", DIM, res);
return res;
}
void sieve_sequential(){
if(DEBUG) printf("! Executing sequential sieve.\n");
double start_time = omp_get_wtime();
int limit = isqrt(DIM);
for(int i = 2; i < limit; i++){
if(DEBUG) printf("\tIterating at %d\n", i);
if(a_seq[i] == 0){
int count = 0;
for(int j = i*i+i*count; i*i+i*count < DIM; count++){
a_seq[i*i+i*count] = 1;
if(DEBUG) printf("\t\t%d is not prime.\n", i*i+i*count );
}
}
}
double time = omp_get_wtime() - start_time;
printf("! Executed sequential sieve: %f\n", time);
}
void sieve_parallel(){
double start_time = omp_get_wtime();
int limit = isqrt(DIM);
#pragma omp parallel
{
int id = omp_get_thread_num(); 
int nthreads = omp_get_num_threads();
printf("! Running %d of %d threads\n", id, nthreads);
for(int i = id+2; i < limit; i = i + nthreads){
if(DEBUG) printf("\tIterating at %d\n", i);
if(a_par[i] == 0){
int count = 0;
for(int j = i*i+i*count; i*i+i*count < DIM; count++){
a_par[i*i+i*count] = 1;
if(DEBUG) printf("\t\t%d is not prime.\n", i*i+i*count );
}
}
}
}
double time = omp_get_wtime() - start_time;
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Sieve of Erastothenes, %d, %d, %f\n", DIM, THREADS, time);
fclose(fp);
printf("! Executed parallelized sieve: %f\n", time);
}
int test_equality(){
if(DEBUG) printf("! Asserting equality of results\n");
if(DEBUG) printf("\ta_seq\t|\ta_par\n");
for(int i = 0; i < DIM; i++){
if(DEBUG) printf("%d\t%d\t|\t %d\n", i, a_seq[i], a_par[i]);
if(a_seq[i] != a_par[i]){
return 1;
}
}
return 0;
}
