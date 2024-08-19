#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <omp.h>
typedef enum Ordering {ASCENDING, DESCENDING, RANDOM} Order;
int debug = 0;
void print_v(int *v, long l) {
printf("\n");
for(long i = 0; i < l; i++) {
if(i != 0 && (i % 10 == 0)) {
printf("\n");
}
printf("%d ", v[i]);
}
printf("\n");
}
int check_result(int *v, long l) {
int prev = v[0];
for(long i = 1; i < l; i++) {
if(prev > v[i]) {
return 0;
}
prev = v[i];
}
return 1;
}
void print_results(struct timeval tv1, struct timeval tv2, long length, int *v) {
double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
(double) (tv2.tv_sec - tv1.tv_sec);
printf("Output (%s):\n\n %10s %13s %13s\n",
check_result(v, length) == 0 ? "incorrect" : "correct", "Elements", "Time in s", "Elements/s");
printf("%10zu % .6e % .6e\n",
length,
time,
(double) length / time);
}
void merge(int *a, long low, long mid, long high, int *b) {
long i = low, j = mid;
for(long k = low; k < high; k++) {
if(i < mid && (j >= high || a[i] <= a[j])) {
b[k] = a[i];
i++;
} else {
b[k] = a[j];
j++;
}
}
}
void split_seq(int *b, long low, long high, int *a) {
if(high - low < 2)
return;
long mid = (low + high) >> 1; 
split_seq(a, low, mid, b); 
split_seq(a, mid, high, b); 
merge(b, low, mid, high, a);
}
void split_parallel(int *b, long low, long high, int *a) {
if(high - low < 1000) {
split_seq(b, low, high, a);
return;
}
long mid = (low + high) >> 1; 
#pragma omp task shared(a, b) firstprivate(low, mid)
split_parallel(a, low, mid, b); 
#pragma omp task shared(a, b) firstprivate(mid, high)
split_parallel(a, mid, high, b); 
#pragma omp taskwait
merge(b, low, mid, high, a);
}
void msort_parallel(int *v, long l, int threads){
struct timeval tv1, tv2;
int *b = (int*)malloc(l*sizeof(int));
memcpy(b, v, l*sizeof(int));
omp_set_num_threads(threads);
printf("Running in parallel with OpenMP. No of threads: %d \n", omp_get_max_threads());
gettimeofday(&tv1, NULL);
#pragma omp parallel shared(v, l, b) num_threads(threads)
{
#pragma omp single
{
split_parallel(b, 0, l, v);
};
}
gettimeofday(&tv2, NULL);
print_results(tv1, tv2, l, v);
}
void msort_seq(int *v, long l) {
struct timeval tv1, tv2;
int *b = (int*)malloc(l*sizeof(int));
memcpy(b, v, l*sizeof(int));
printf("Running sequential\n");
gettimeofday(&tv1, NULL);
split_seq(b, 0, l, v);
gettimeofday(&tv2, NULL);
print_results(tv1, tv2, l, v);
}
int main(int argc, char **argv) {
int c;
int seed = 42;
long length = 1e8;
Order order = ASCENDING;
int *vector;
int sequential = 0;
int threads = 1;
while((c = getopt(argc, argv, "adrgl:s:St:")) != -1) {
switch(c) {
case 'a':
order = ASCENDING;
break;
case 'd':
order = DESCENDING;
break;
case 'r':
order = RANDOM;
break;
case 'l':
length = atol(optarg);
break;
case 'g':
debug = 1;
break;
case 's':
seed = atoi(optarg);
break;
case 'S':
sequential = 1;
break;
case 't':
threads = atoi(optarg);
break;
case '?':
if(optopt == 'l' || optopt == 's') {
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
}
else if(isprint(optopt)) {
fprintf(stderr, "Unknown option '-%c'.\n", optopt);
}
else {
fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
}
return -1;
default:
return -1;
}
}
srand(seed);
vector = (int*)malloc(length*sizeof(int));
if(vector == NULL) {
fprintf(stderr, "Malloc failed...\n");
return -1;
}
switch(order){
case ASCENDING:
for(long i = 0; i < length; i++) {
vector[i] = (int)i;
}
break;
case DESCENDING:
for(long i = 0; i < length; i++) {
vector[i] = (int)(length - i);
} 
break;
case RANDOM:
for(long i = 0; i < length; i++) {
vector[i] = rand();
}
break;
}
if(debug) {
print_v(vector, length);
}
if(sequential) {
msort_seq(vector, length);
} else {
msort_parallel(vector, length, threads);
}
if(debug) {
print_v(vector, length);
}
return 0;
}
