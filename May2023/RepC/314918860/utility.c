#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "omp.h"
#include "utility.h"
void swap(float* xp, float* yp) {
float temp = *xp;
*xp = *yp;
*yp = temp;
}
float* copy_bloc(float** bloc, int bloc_size) {
float* new_b = (float*) malloc(sizeof(float) * bloc_size);
for (size_t i = 0; i < bloc_size; i++)
*(new_b+i) = *(*bloc+i);
return new_b;
}
void selection_sort(float **bloc, int k) {
for (size_t i = 0; i < k; i++) {
int min_idx = i;
for (int j = i + 1; j < k; j++) {
if (*(*bloc+j) < *(*bloc+min_idx))
min_idx = j;
}
swap(*bloc+min_idx, *bloc+i);
}
}
void heapify(float **bloc, int bloc_size, int i) {
int largest = i; 
int l = 2 * i + 1; 
int r = 2 * i + 2; 
if (l < bloc_size && **bloc+l > **bloc+largest)
largest = l;
if (r < bloc_size && **bloc+r > **bloc+largest)
largest = r;
if (largest != i) {
swap(*bloc+i, *bloc+largest);
heapify(bloc, bloc_size, largest);
}
}
void heap_sort(float** bloc, int bloc_size) {
for (int i = bloc_size / 2 - 1; i >= 0; i--)
heapify(bloc, bloc_size, i);
for (int i = bloc_size - 1; i > 0; i--) {
swap(*bloc, *bloc+i);
heapify(bloc, i, 0);
}
}
void tri(float **bloc, int k){
heap_sort(bloc, k);
}
void tri_merge(float **bloc1, float **bloc2, int k) {
float* new_bloc = (float*) malloc(sizeof(float) * k*2);
for (size_t i = 0; i < k*2; i++) {
if (i < k) *(new_bloc+i) = *(*bloc1+i);
else *(new_bloc+i) = *(*bloc2+i-k);
}
tri(&new_bloc, k*2);
for (size_t i = 0; i < k; i++){
*(*bloc1+i) = *(new_bloc+i);
*(*bloc2+i) = *(new_bloc+i+k);
}
free(new_bloc);
}
float* generator(int k) {
float* bloc = (float*) malloc(sizeof(float) * k);
for (size_t i = 0; i < k; i++)
*(bloc+i) = (float)rand()/RAND_MAX;
return bloc;
}
void free_db(float** db, int n) {
for (size_t i = 0; i < n; i++)
free(*(db+i));
free(db);
}
void parallel_sort(float*** db, int n, int k, performance_measures* pm) {
double t1 = omp_get_wtime();
#pragma omp parallel for firstprivate(k)
for (size_t i = 0; i < n; i++) 
tri(*db+i, k);
double t2 = omp_get_wtime();
pm->first_sort = t2-t1;
pm->second_sort = 0;
for (size_t j = 0; j < n; j++) {
int bi = 1 + (j % 2);
t1 = omp_get_wtime();
#pragma omp parallel for firstprivate(bi, n, k)
for (size_t i = 0; i < n/2; i++) {
int b1 = (bi + 2 * i) % n;
int b2 = (bi + 2 * i+1) % n;
int min = MIN(b1, b2);
int max = MAX(b1, b2);
tri_merge(*db+min, *db+max, k);
}
t2 = omp_get_wtime();
pm->second_sort += t2-t1;
}
}
void generate_performances(char* filename, int nk[][2], int nk_size, int nb_threads[], int nb_threads_size) {
printf("Running tests...\n");
FILE* fp = NULL;
fp = fopen(filename, "w");
fprintf(fp, "N,K,NxK,#Threads,FirstSort,SecondSort,Performance\n");
for (size_t i = 0; i < nk_size; i++) {
int n = nk[i][0];
int k = nk[i][1];
printf("\n%5s | %5s | %8s | %7s | %13s | %13s | %13s\n",
"N", "K", "NxK", "Threads", "1st Sort", "2nd Sort", "Total");
for (size_t j = 0; j < nb_threads_size; j++) {
int nb_th = nb_threads[j];
printf("%5d | %5d | %8d | %7d | %13s | %13s | %13s", 
n, k, n*k, nb_th, "CALCULATING..", "CALCULATING..", "CALCULATING..");
fflush(stdout);
performance_measures pm = get_performance_measures(n, k, nb_th);
printf("\33[2K\r%5d | %5d | %8d | %7d | %13f | %13f | %13f\n", 
n, k, n*k, nb_th, pm.first_sort, pm.second_sort, pm.sorting_span);
fprintf(fp, "%d,%d,%d,%d,%f,%f,%f\n", 
n, k, n*k, nb_th, pm.first_sort, pm.second_sort, pm.sorting_span);
fflush(fp); 
}
printf("\n");
}
fclose(fp);
printf("%s file has been created successfully!\n", filename);
}
performance_measures get_performance_measures(int n, int k, int nb_threads) {
performance_measures pm;
srand((unsigned) time(NULL));
omp_set_num_threads(nb_threads);
double t1 = omp_get_wtime();
float** db = (float**) malloc(sizeof(float*) * n);
for (size_t i = 0; i < n; i++) {
*(db+i) = NULL;
*(db+i) = generator(k);        
}
float** sorted_db = (float**) malloc(sizeof(float*) * n);
for (size_t i = 0; i < n; i++) {
float* bloc = (float*) malloc(sizeof(float) * k);
for (size_t j = 0; j < k; j++) 
*(bloc+j) = db[i][j];
*(sorted_db+i) = bloc;
}
double t2 = omp_get_wtime();
pm.generating_span = t2 - t1;
t1 = omp_get_wtime();
parallel_sort(&sorted_db, n, k, &pm);
t2 = omp_get_wtime();
pm.sorting_span = t2 - t1;
pm.total_span = pm.generating_span + pm.sorting_span;
#if DEBUG==1
printf("UNSORTED DATABASE\n");
d_dump_db(db, n, k);
printf("-----------------------------------\n");
printf("SORTED DATABASE\n");
d_dump_db(sorted_db, n, k);
printf("\n\n");
#endif
free_db(sorted_db, n);
free_db(db, n);
return pm;
}
void d_dump_db(float** db, int n, int k) {
for (size_t i = 0; i < n; i++) {
printf("db[%ld]\n", i);
for (size_t j = 0; j < k; j++) {
printf("%.2f|", *(*(db+i)+j));
}
printf("\n\n");
}
}
int strpos(char *haystack, char *needle) {
char *p = strstr(haystack, needle);
if (p)
return p - haystack;
return -1;
}