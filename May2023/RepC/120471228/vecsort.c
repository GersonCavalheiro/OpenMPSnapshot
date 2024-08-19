#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <ctype.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>
int TASK_THREADS = 6;
int DATA_THREADS = 2;
typedef enum Ordering {
ASCENDING, DESCENDING, RANDOM
} Order;
void print_v(int *v, long l) {
printf("\n");
for (long i = 0; i < l; i++) {
if (i != 0 && (i % 10 == 0)) {
printf("\n");
}
printf("%d ", v[i]);
}
printf("\n");
}
void print_2d_v(int **v, long rows, const long* row_lengths) {
printf("\n");
for (long i = 0; i < rows; i++) {
long length = row_lengths[i];
for (long j = 0; j < length; j++) {
printf("%d \t", v[i][j]);
}
printf("\n");
}
}
int check_result(int **v, long rows, const long* row_lengths) {
for (long r = 0; r < rows; r++) {
int prev = v[r][0];
long l = row_lengths[r];
for (long i = 1; i < l; i++) {
if (prev > v[r][i]) {
printf("warning: vector at row[%ld] is not sorted", r);
print_v(v[r], l);
return 0;
}
prev = v[r][i];
}
}
return 1;
}
void print_results(struct timeval tv1, struct timeval tv2, long rows, long sum_elements, long* row_lengths, int **v) {
double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
(double) (tv2.tv_sec - tv1.tv_sec);
printf("Output \033[32;1m(%s)\033[0m:\n\n %10s %13s %13s\n ",
check_result(v, rows, row_lengths) == 0 ? "incorrect" : "correct", " Elements", "Time in s", "Elements/s");
printf("%10zu % .6e % .6e\n",
sum_elements,
time,
(double) sum_elements / time);
}
void merge(const int *a, long low, long mid, long high, int *b) {
long i = low, j = mid;
for (long k = low; k < high; k++) {
if (i < mid && (j >= high || a[i] <= a[j])) {
b[k] = a[i];
i++;
} else {
b[k] = a[j];
j++;
}
}
}
void split_seq(int *b, long low, long high, int *a) {
if (high - low < 2)
return;
long mid = (low + high) >> 1; 
split_seq(a, low, mid, b); 
split_seq(a, mid, high, b); 
merge(b, low, mid, high, a);
}
void split_parallel(int *b, long low, long high, int *a) {
if (high - low < 1000) {
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
void vecsort_datapar(int **vector, long rows, long* row_lengths, long sum_elements) {
struct timeval tv1, tv2;
printf("Running parallel - Rows %ld \n", rows);
printf("Number of threads for data par %d\n", DATA_THREADS);
int **b = (int **) malloc(sizeof(int) * sum_elements);
long row;
for (row = 0; row < rows; row++) {
long length = row_lengths[row];
b[row] = (int *) malloc(sizeof(int) * length);
memcpy(b[row], vector[row], length * sizeof(int));
}
gettimeofday(&tv1, NULL);
#pragma omp parallel for num_threads(DATA_THREADS)
for (row = 0; row < rows; row++) {
long length = row_lengths[row];
split_seq(b[row], 0, length, vector[row]);
}
gettimeofday(&tv2, NULL);
print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}
void vecsort_taskpar(int **vector, long rows, long* row_lengths, long sum_elements) {
struct timeval tv1, tv2;
printf("Running parallel - Rows %ld \n", rows);
printf("Number of threads for data par %d\n", DATA_THREADS);
printf("Number of threads for task par %d\n", TASK_THREADS);
omp_set_nested(1);
int **b = (int **) malloc(sizeof(int) * sum_elements);
for (long row = 0; row < rows; row++) {
long length = row_lengths[row];
b[row] = (int *) malloc(sizeof(int) * length);
memcpy(b[row], vector[row], length * sizeof(int));
}
gettimeofday(&tv1, NULL);
#pragma omp parallel num_threads(DATA_THREADS)
{
#pragma omp for
for (long row = 0; row < rows; row++) {
long length = row_lengths[row];
#pragma omp parallel num_threads(TASK_THREADS)
{
#pragma omp single
{
split_parallel(b[row], 0, length, vector[row]);
};
}
}
}
gettimeofday(&tv2, NULL);
print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}
void vecsort_seq(int **vector, long rows, long* row_lengths, long sum_elements) {
struct timeval tv1, tv2;
printf("Running sequential - %ld Rows\n", rows);
int **b = (int **) malloc(sizeof(int) * sum_elements);
long row;
for (row = 0; row < rows; row++) {
long length = row_lengths[row];
b[row] = (int *) malloc(sizeof(int) * length);
memcpy(b[row], vector[row], length * sizeof(int));
}
gettimeofday(&tv1, NULL);
for (row = 0; row < rows; row++) {
long length = row_lengths[row];
split_seq(b[row], 0, length, vector[row]);
}
gettimeofday(&tv2, NULL);
print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}
int main(int argc, char **argv) {
int c;
int seed = 42;
long length = 1e4;
long rows = 1e2;
int var_length = 0;
long *row_lengths;
long sum_elements = 0;
int sequential = 0;
int datapar_only = 0;
int debug = 0;
Order order = ASCENDING;
while ((c = getopt(argc, argv, "adrgl:s:R:D:T:SPv")) != -1) {
switch (c) {
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
case 'R':
rows = atoi(optarg);
break;
case 'S':
sequential = 1;
break;
case 'P':
datapar_only = 1;
break;
case 'v':
var_length = 1;
break;
case 'D':
DATA_THREADS = atoi(optarg);
break;
case 'T':
TASK_THREADS = atoi(optarg);
break;
case '?':
if (optopt == 'l' || optopt == 's') {
fprintf(stderr, "Option -%c requires an argument.\n", optopt);
} else if (isprint(optopt)) {
fprintf(stderr, "Unknown option '-%c'.\n", optopt);
} else {
fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
}
return -1;
default:
return -1;
}
}
row_lengths = (long*) malloc(rows * sizeof(long));
if ( var_length ) {
printf("Variable length mode.\n");
for (int r = 0; r < rows; r++) {
long l = rand() % (length + 1 - (length / 2)) + (length / 2);
row_lengths[r] = l;
sum_elements += l;
}
} else {
printf("Fixed size mode.\n");
for (int r = 0; r < rows; r++) {
row_lengths[r] = length;
sum_elements += length;
}
}
printf("A total of %ld elements will be sorted.\n", sum_elements);
int **vector = (int **) malloc(sum_elements * sizeof(int));
if (vector == NULL) {
printf("Malloc failed...");
return -1;
}
for (int r = 0; r < rows; r++) {
srand(seed + r);
long l = row_lengths[r];
int *array;
array = (int *) malloc(l * sizeof(int));
if (array == NULL) {
fprintf(stderr, "Malloc failed...\n");
return -1;
}
switch (order) {
case ASCENDING:
for (long i = 0; i < l; i++) {
array[i] = (int) i;
}
break;
case DESCENDING:
for (long i = 0; i < l; i++) {
array[i] = (int) (l - i);
}
break;
case RANDOM:
for (long i = 0; i < l; i++) {
array[i] = rand();
}
break;
}
vector[r] = array;
}
if (debug) {
printf("Initial vector ::");
print_2d_v(vector, rows, row_lengths);
}
if (sequential) {
vecsort_seq(vector, rows, row_lengths, sum_elements);
} else {
if (datapar_only) {
vecsort_datapar(vector, rows, row_lengths, sum_elements);
} else {
vecsort_taskpar(vector, rows, row_lengths, sum_elements);
}
}
if (debug) {
printf("Final vector ::");
print_2d_v(vector, rows, row_lengths);
}
return 0;
}
