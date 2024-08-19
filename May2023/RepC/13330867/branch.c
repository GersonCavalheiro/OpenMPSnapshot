#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define VECTOR_STOP       66
#define VECTOR_GO         77
#define NO_VECTOR         88
#define INS_HEAVY         99
#define WITH_BRANCHES      1
#define WITHOUT_BRANCHES   0
extern int fill_vec(int *vector, int vector_length, int iterations, int branch, 
int *nfunc, int *rank);
int main(int argc, char ** argv)
{
int      my_ID;           
int      vector_length;   
int      nfunc;           
int      rank;            
double   branch_time,     
no_branch_time;
double   ops;             
int      iterations;      
int      i, iter, aux;    
char     *branch_type;    
int      btype;           
int      total=0, 
total_ref;       
int      nthread_input;   
int      nthread; 
int      num_error=0;     
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Branching Bonanza\n");
if (argc != 5){
printf("Usage:     %s <# threads> <# iterations> <vector length>", *argv);
printf("<branching type>\n");
printf("branching type: vector_go, vector_stop, no_vector, ins_heavy\n");
exit(EXIT_FAILURE);
}
nthread_input = atoi(*++argv);
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
iterations = atoi(*++argv);
if (iterations < 1 || iterations%2==1){
printf("ERROR: Iterations must be positive and even : %d \n", iterations);
exit(EXIT_FAILURE);
}
vector_length  = atoi(*++argv);
if (vector_length < 1){
printf("ERROR: loop length must be >= 1 : %d \n",vector_length);
exit(EXIT_FAILURE);
}
branch_type = *++argv;
if      (!strcmp(branch_type,"vector_stop")) btype = VECTOR_STOP;
else if (!strcmp(branch_type,"vector_go"  )) btype = VECTOR_GO;
else if (!strcmp(branch_type,"no_vector"  )) btype = NO_VECTOR;
else if (!strcmp(branch_type,"ins_heavy"  )) btype = INS_HEAVY;
else  {
printf("Wrong branch type: %s; choose vector_stop, vector_go, ", branch_type);
printf("no_vector, or ins_heavy\n");
exit(EXIT_FAILURE);
}
#pragma omp parallel private(i, my_ID, iter, aux, nfunc, rank) reduction(+:total)
{
int * RESTRICT vector; int * RESTRICT index;
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
printf("Number of threads          = %d\n", nthread_input);
printf("Vector length              = %d\n", vector_length);
printf("Number of iterations       = %d\n", iterations);
printf("Branching type             = %s\n", branch_type);
#if RESTRICT_KEYWORD
printf("No aliasing                = on\n");
#else
printf("No aliasing                = off\n");
#endif
}
}
bail_out(num_error);
my_ID = omp_get_thread_num();
vector = prk_malloc(vector_length*2*sizeof(int));
if (!vector) {
printf("ERROR: Thread %d failed to allocate space for vector\n", my_ID);
num_error = 1;
}
bail_out(num_error);
index   = vector + vector_length;
for (i=0; i<vector_length; i++) { 
vector[i]  = 3 - (i&7);
index[i]   = i;
}
#pragma omp barrier   
#pragma omp master
{   
branch_time = wtime();
}
switch (btype) {
case VECTOR_STOP:
for (iter=0; iter<iterations; iter+=2) {
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) { 
aux = -(3 - (i&7));
if (vector[index[i]]>0) vector[i] -= 2*vector[i];
else                    vector[i] -= 2*aux;
}
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) { 
aux = (3 - (i&7));
if (vector[index[i]]>0) vector[i] -= 2*vector[i];
else                    vector[i] -= 2*aux;
}
}
break;
case VECTOR_GO:
for (iter=0; iter<iterations; iter+=2) {
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = -(3 - (i&7));
if (aux>0) vector[i] -= 2*vector[i];
else       vector[i] -= 2*aux;
}
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = (3 - (i&7));
if (aux>0) vector[i] -= 2*vector[i];
else       vector[i] -= 2*aux;
}
}
break;
case NO_VECTOR:
for (iter=0; iter<iterations; iter+=2) {
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = -(3 - (i&7));
if (aux>0) vector[i] -= 2*vector[index[i]];
else       vector[i] -= 2*aux;
}
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = (3 - (i&7));
if (aux>0) vector[i] -= 2*vector[index[i]];
else       vector[i] -= 2*aux;
}
}
break;
case INS_HEAVY:
fill_vec(vector, vector_length, iterations, WITH_BRANCHES, &nfunc, &rank);
}
#pragma omp master
{
branch_time = wtime() - branch_time;
if (btype == INS_HEAVY) {
printf("Number of matrix functions = %d\n", nfunc);
printf("Matrix order               = %d\n", rank);
}
}
#pragma omp barrier
#pragma omp master
{   
no_branch_time = wtime();
}
switch (btype) {
case VECTOR_STOP:
case VECTOR_GO:
for (iter=0; iter<iterations; iter+=2) {
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) { 
aux = -(3-(i&7)); 
vector[i] -= (vector[i] + aux);
}
for (i=0; i<vector_length; i++) {
aux = (3-(i&7)); 
vector[i] -= (vector[i] + aux);
}
}
break;
case NO_VECTOR:
for (iter=0; iter<iterations; iter+=2) {
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = -(3-(i&7));
vector[i] -= (vector[index[i]]+aux); 
}
PRAGMA_OMP_FOR_SIMD
for (i=0; i<vector_length; i++) {
aux = (3-(i&7));
vector[i] -= (vector[index[i]]+aux); 
}
}
break;
case INS_HEAVY:
fill_vec(vector, vector_length, iterations, WITHOUT_BRANCHES, &nfunc, &rank);
}
#pragma omp master
{
no_branch_time = wtime() - no_branch_time;
ops = (double)vector_length * (double)iterations * (double)nthread;
if (btype == INS_HEAVY) ops *= rank*(rank*19 + 6);
else                    ops *= 4;
}
for (total = 0, i=0; i<vector_length; i++) total += vector[i];
} 
total_ref = ((vector_length%8)*(vector_length%8-8) + vector_length)/2*nthread;
if (total == total_ref) {
printf("Solution validates\n");
printf("Rate (Mops/s) with branches:    %lf time (s): %lf\n", 
ops/(branch_time*1.e6), branch_time);
printf("Rate (Mops/s) without branches: %lf time (s): %lf\n", 
ops/(no_branch_time*1.e6), no_branch_time);
#if VERBOSE
printf("Array sum = %d, reference value = %d\n", total, total_ref);
#endif     
}
else {
printf("ERROR: array sum = %d, reference value = %d\n", total, total_ref);
}
exit(EXIT_SUCCESS);
}
