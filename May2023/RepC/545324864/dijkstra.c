#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>
#define DEBUG 0
#ifndef THREADS
#define THREADS 4
#endif
#ifndef DIM
#define DIM 10
#endif
FILE *fp;
int a_seq[DIM][DIM];
int a_par[DIM][DIM];
int dist[DIM];
bool vis[DIM];
void initialize() {
srand(time(0));
for (int i = 0; i < DIM; i++) {
for (int j = i; j < DIM; j++) {
if (i - j == 0 || rand() % 5 == 0) { 
a_seq[j][i] = a_seq[i][j] = 0;
a_par[j][i] = a_par[i][j] = 0;
}  else {
int r = rand() % 100;
a_seq[j][i] = a_seq[i][j] = r;  
a_par[j][i] = a_par[i][j] = r; 
}
}
}
}
int minDistance(int const dist[], bool const prev[]){
int min = INT_MAX;
int min_index;
for (int v = 0; v < DIM; v++){
if (!prev[v] && dist[v] <= min){
min = dist[v];
min_index = v;
}
}
return min_index;
}
void dijkstra_sequential(){
int total_distance = 0;
for (int i = 0; i < DIM; i++){
dist[i] = INT_MAX;
vis[i] = false;
}
dist[0] = 0;
for (int count = 0; count < DIM - 1; count++) {
int u = minDistance(dist, vis);
vis[u] = true;
for (int v = 0; v < DIM; v++){
if (!vis[v] && a_seq[u][v] && dist[u] != INT_MAX && dist[u] + a_seq[u][v] < dist[v]){
dist[v] += a_seq[u][v];
total_distance += a_seq[u][v];
}
}
}
}
void dijkstra_parallel(){
int total_distance = 0;
for (int i = 0; i < DIM; i++){
dist[i] = INT_MAX;
vis[i] = false;
}
dist[0] = 0;
for (int count = 0; count < DIM - 1; count++) {
int u = minDistance(dist, vis);
vis[u] = true;
#pragma omp parallel for schedule(static) default(none) shared(dist, vis, a_par, total_distance) firstprivate(u)
for (int v = 0; v < DIM; v++){
if (!vis[v] && a_par[u][v] && dist[u] != INT_MAX && dist[u] + a_par[u][v] < dist[v]){
dist[v] += a_par[u][v];
total_distance += a_par[u][v];
}
}
}
}
int test_equality()
{
if (DEBUG) printf("! Asserting equality\n");
if (DEBUG) printf("\ta_seq\t|\ta_par\n");
for (int i = 0; i < DIM; i++)
{
for(int j = 0; j < DIM; j++){
if (DEBUG) printf("[%d,%d]\t%d\t|\t %d\n", i,j, a_seq[i][j], a_par[i][j]);
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
double start_time, time;
start_time = omp_get_wtime();
dijkstra_sequential();
time = omp_get_wtime() - start_time;
printf("! Executed sequential dijkstra: %f\n", time);
start_time = omp_get_wtime();
dijkstra_parallel();
time = omp_get_wtime() - start_time;
printf("! Executed parallelized dijkstra: %f\n", time);
int res = test_equality();
printf("! Dijkstra exit: %d\n", res);
if (res != 0)
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Dijkstra error: Parallel and Sequential results differ.");
fclose(fp);
}
else
{
fp = fopen("log/c_std.csv", "a");
fprintf(fp, "Dijkstra, %d, %d, %f\n", DIM, THREADS, time);
fclose(fp);
}
return 0;
}