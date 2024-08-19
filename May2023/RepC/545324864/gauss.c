#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define DEBUG 0
#ifndef DIM
#define DIM 3
#endif
#ifndef THREADS
#define THREADS 4
#endif
FILE *fp;
double a_seq[DIM][DIM];
double a_par[DIM][DIM];
void dswap(double *a, double *b) {
double t = *a;
*a = *b;
*b = t;
}
void initialize() {
for(int i = 0; i < DIM; i++){
for(int j = 0;  j < DIM; j++){
int r  = rand() % 100;
a_seq[i][j] = r;
a_par[i][j] = r;
}
}
}
void gauss_sequential(){
int h = 0;
int k = 0;
while (h < DIM && k < DIM) {
int i_max = h;
for (int i = h; i < DIM; i++) {
if (fabs(a_seq[i_max][k]) < fabs(a_seq[i][k])) {
i_max = i;
}
}
if (a_seq[i_max][k] == 0) {
k++;
continue;
}
for (int i = 0; i < DIM; i++) {
dswap(&a_seq[h][i], &a_seq[i_max][i]);
}
for (int i = h + 1; i < DIM; i++) {
double f = (double) a_seq[i][k] / a_seq[h][k];
a_seq[i][k] = 0;
for (int j = k + 1; j < DIM; j++) {
a_seq[i][j] -= a_seq[h][j] * f;
}
}
h++;
k++;
}
}
void gauss_parallel() {
int h = 0;
int k = 0;
while (h < DIM && k < DIM) {
int i_max = h;
for (int i = h; i < DIM; i++) {
if (fabs(a_par[i_max][k]) < fabs(a_par[i][k])) {
i_max = i;
}
}
if (a_par[i_max][k] == 0) {
k++;
continue;
}
for (int i = 0; i < DIM; i++) {
dswap(&a_par[h][i], &a_par[i_max][i]);
}
#pragma omp parallel for schedule(static) default(none) shared(a_par) firstprivate(h, k)
for (int i = h + 1; i < DIM; i++) {
double f = (double) a_par[i][k] / a_par[h][k];
a_par[i][k] = 0;
for (int j = k + 1; j < DIM; j++) {
a_par[i][j] -= a_par[h][j] * f;
}
}
h++;
k++;
}
}
int test_equality()
{
if (DEBUG) printf("! Asserting equality of results\n");
if (DEBUG) printf("\ta_seq\t|\ta_par\n");
for (int i = 0; i < DIM; i++)
{
for(int j = 0; j < DIM; j++){
if (DEBUG) printf("[%d,%d]\t%f\t|\t %f\n", i,j, a_seq[i][j], a_par[i][j]);
if (a_seq[i][j] != a_par[i][j])
{
return 1;
}
}
}
return 0;
}
int main() {
omp_set_num_threads(THREADS);
printf("! Utilizing %d threads over %d elements\n", THREADS, DIM);
initialize();
srand(time(0));
double start_time, time;
start_time = omp_get_wtime();
gauss_sequential();
time = omp_get_wtime() - start_time;
printf("! Executed sequential Gauss: %f\n", time);
start_time = omp_get_wtime();
gauss_parallel();
time = omp_get_wtime() - start_time;
printf("! Executed parallelized Gauss: %f\n", time);
int res = test_equality();
printf("! Gauss exit: %d\n", res);
if (res != 0)
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Gauss error: Parallel and Sequential results differ.");
fclose(fp);
}
else
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Gauss, %d, %d, %f\n", DIM, THREADS, time);
fclose(fp);
}
return 0;
}