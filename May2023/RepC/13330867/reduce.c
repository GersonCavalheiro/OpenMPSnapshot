#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define LINEAR            11
#define BINARY_BARRIER    12
#define BINARY_P2P        13
#define LONG_OPTIMAL      14
#define NONE              15
#define LOCAL             16
#define VEC0(id,i)        vector[(id        )*(vector_length)+i]
#define VEC1(id,i)        vector[(id+nthread)*(vector_length)+i]
#define LINEWORDS         16
#define flag(i)           flag[(i)*LINEWORDS]
int main(int argc, char ** argv)
{
int    my_ID;           
long   vector_length;   
long   total_length;    
double reduce_time,     
avgtime;
double epsilon=1.e-8;   
int    group_size,      
old_size,        
i, id, iter, stage; 
double element_value;   
char   *algorithm;      
int    intalgorithm;    
int    iterations;      
int    flag[MAX_THREADS*LINEWORDS]; 
int    start[MAX_THREADS],
end[MAX_THREADS];
long   segment_size;
int    my_donor, my_segment;
int    nthread_input,   
nthread;   
double * RESTRICT vector;
int    num_error=0;     
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Vector Reduction\n");
if (argc != 4 && argc != 5){
printf("Usage:     %s <# threads> <# iterations> <vector length> ", *argv);
printf("[<alghorithm>]\n");
printf("Algorithm: linear, binary-barrier, binary-p2p, or long-optimal\n");
return(EXIT_FAILURE);
}
nthread_input = atoi(*++argv); 
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
iterations = atoi(*++argv);
if (iterations < 1){
printf("ERROR: Iterations must be positive : %d \n", iterations);
exit(EXIT_FAILURE);
}
vector_length  = atol(*++argv);
if (vector_length < 1){
printf("ERROR: vector length must be >= 1 : %ld \n",vector_length);
exit(EXIT_FAILURE);
}
total_length = vector_length*2*nthread_input*sizeof(double);
vector = (double *) prk_malloc(total_length);
if (!vector) {
printf("ERROR: Could not allocate space for vectors: %ld\n", total_length);
exit(EXIT_FAILURE);
}
algorithm = "binary-p2p";
if (argc == 5) algorithm = *++argv;
intalgorithm = NONE;
if (!strcmp(algorithm,"linear"        )) intalgorithm = LINEAR;
if (!strcmp(algorithm,"binary-barrier")) intalgorithm = BINARY_BARRIER;
if (!strcmp(algorithm,"binary-p2p"    )) intalgorithm = BINARY_P2P;
if (!strcmp(algorithm,"long-optimal"  )) intalgorithm = LONG_OPTIMAL;
if (intalgorithm == NONE) {
printf("Wrong algorithm: %s; choose linear, binary-barrier, ", algorithm);
printf("binary-p2p, or long-optimal\n");
exit(EXIT_FAILURE);
}
else {
if (nthread_input == 1) intalgorithm = LOCAL;
}
#pragma omp parallel private(i, old_size, group_size, my_ID, iter, start, end, segment_size, stage, id, my_donor, my_segment) 
{
my_ID = omp_get_thread_num();
#pragma omp master 
{
nthread = omp_get_num_threads();
if (nthread != nthread_input) {
num_error = 1;
printf("ERROR: number of requested threads %d does not equal ",
nthread_input);
printf("number of spawned threads %d\n", nthread);
} 
else {
printf("Number of threads              = %d\n",nthread_input);
printf("Vector length                  = %ld\n", vector_length);
printf("Reduction algorithm            = %s\n", algorithm);
printf("Number of iterations           = %d\n", iterations);
}
}
bail_out(num_error);
for (iter=0; iter<=iterations; iter++) {
if (iter == 1) { 
#pragma omp barrier
#pragma omp master
{
reduce_time = wtime();
}
}
if (intalgorithm == LONG_OPTIMAL) {
#pragma omp barrier
}
for (i=0; i<vector_length; i++) {
VEC0(my_ID,i) = (double)(my_ID+1);
VEC1(my_ID,i) = (double)(my_ID+1+nthread);
}
if (intalgorithm == BINARY_P2P) {
#pragma omp barrier
flag(my_ID) = 0;
#pragma omp barrier
}    
for (i=0; i<vector_length; i++) {
VEC0(my_ID,i) += VEC1(my_ID,i);
}
switch (intalgorithm) {
case LOCAL:  
break;
case LINEAR:
{
#pragma omp barrier
#pragma omp master
{
for (id=1; id<nthread; id++) {
for (i=0; i<vector_length; i++) {
VEC0(0,i) += VEC0(id,i);
}
}
}
}
break;
case BINARY_BARRIER:
group_size = nthread;
while (group_size >1) {
#pragma omp barrier
old_size = group_size;
group_size = (group_size+1)/2;
if (my_ID < group_size && my_ID+group_size<old_size) {
for (i=0; i<vector_length; i++) {
VEC0(my_ID,i) += VEC0(my_ID+group_size,i);
}
}
}
break;
case BINARY_P2P:
group_size = nthread;
while (group_size >1) {
old_size = group_size;
group_size = (group_size+1)/2;
if (my_ID < group_size && my_ID+group_size<old_size) {
while (flag(my_ID+group_size) == 0) {
#pragma omp flush
}
#pragma omp flush 
for (i=0; i<vector_length; i++) {
VEC0(my_ID,i) += VEC0(my_ID+group_size,i);
}
}
else {
if (my_ID < old_size) {
flag(my_ID) = 1;
#pragma omp flush
}
}
}
break;
case LONG_OPTIMAL:
segment_size = (vector_length+nthread-1)/nthread;
for (id=0; id<nthread; id++) {
start[id] = segment_size*id;
end[id]   = MIN(vector_length,segment_size*(id+1));
}
my_donor   = (my_ID-1+nthread)%nthread;
for (stage=1; stage<nthread; stage++) {
#pragma omp barrier
my_segment = (my_ID-stage+nthread)%nthread;
for (i=start[my_segment]; i<end[my_segment]; i++) {
VEC0(my_ID,i) += VEC0(my_donor,i);
}
}
my_segment = (my_ID+1)%nthread;
if (my_ID != 0)
for (i=start[my_segment]; i<end[my_segment]; i++) {
VEC0(0,i) = VEC0(my_ID,i);
}
break;
} 
} 
#pragma omp barrier
#pragma omp master
{
reduce_time = wtime() - reduce_time;
}
} 
element_value = (double)nthread*(2.0*(double)nthread+1.0);
for (i=0; i<vector_length; i++) {
if (ABS(VEC0(0,i) - element_value) >= epsilon) {
printf("First error at i=%d; value: %lf; reference value: %lf\n",
i, VEC0(0,i), element_value);
exit(EXIT_FAILURE);
}
}
printf("Solution validates\n");
#if VERBOSE
printf("Element verification value: %lf\n", element_value);
#endif
avgtime = reduce_time/iterations;
printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
1.0E-06 * (2.0*nthread-1.0)*vector_length/avgtime, avgtime);
exit(EXIT_SUCCESS);
}
