#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define LIN(i,j) (i+((j)<<lsize))
#if SCRAMBLE
#define REVERSE(a,b)  reverse((a),(b))
#else
#define REVERSE(a,b) (a)
#endif
#define BITS_IN_BYTE 8
static u64Int reverse(register u64Int, int);
static int compare(const void *el1, const void *el2);
int main(int argc, char **argv){
int               iter, r;    
int               lsize;      
int               lsize2;     
int               size;       
s64Int            size2;      
int               radius,     
stencil_size;
s64Int            row, col, first, last; 
s64Int            i, j;       
int               iterations; 
s64Int            elm;        
s64Int            nent;       
double            sparsity;   
double            sparse_time,
avgtime;
double * RESTRICT matrix;     
double * RESTRICT vector;     
double * RESTRICT result;     
double            temp;       
double            vector_sum; 
double            reference_sum; 
double            epsilon = 1.e-8; 
s64Int * RESTRICT colIndex;   
int               nthread_input,  
nthread;
int               num_error=0; 
size_t            vector_space, 
matrix_space,
index_space;
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Sparse matrix-vector multiplication\n");
if (argc != 5) {
printf("Usage: %s <# threads> <# iterations> <2log grid size> <stencil radius>\n",*argv);
exit(EXIT_FAILURE);
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
lsize = atoi(*++argv);
lsize2 = 2*lsize;
size = 1<<lsize;
if (lsize <0) {
printf("ERROR: Log of grid size must be greater than or equal to zero: %d\n",
(int) lsize);
exit(EXIT_FAILURE);
}
size2 = size*size;
radius = atoi(*++argv);
if (radius <0) {
printf("ERROR: Stencil radius must be non-negative: %d\n", (int) size);
exit(EXIT_FAILURE);
}
if (size <2*radius+1) {
printf("ERROR: Grid extent %d smaller than stencil diameter 2*%d+1= %d\n",
size, radius, radius*2+1);
exit(EXIT_FAILURE);
}
stencil_size = 4*radius+1;
sparsity = (double)(4*radius+1)/(double)size2;
nent = size2*stencil_size;
matrix_space = nent*sizeof(double);
if (matrix_space/sizeof(double) != nent) {
printf("ERROR: Cannot represent space for matrix: %zu\n", matrix_space);
exit(EXIT_FAILURE);
}
matrix = (double *) prk_malloc(matrix_space);
if (!matrix) {
printf("ERROR: Could not allocate space for sparse matrix: "FSTR64U"\n", nent);
exit(EXIT_FAILURE);
}
vector_space = 2*size2*sizeof(double);
if (vector_space/sizeof(double) != 2*size2) {
printf("ERROR: Cannot represent space for vectors: %zu\n", vector_space);
exit(EXIT_FAILURE);
}
vector = (double *) prk_malloc(vector_space);
if (!vector) {
printf("ERROR: Could not allocate space for vectors: %d\n", (int)(2*size2));
exit(EXIT_FAILURE);
}
result = vector + size2;
index_space = nent*sizeof(s64Int);
if (index_space/sizeof(s64Int) != nent) {
printf("ERROR: Cannot represent space for column indices: %zu\n", index_space);
exit(EXIT_FAILURE);
}
colIndex = (s64Int *) prk_malloc(index_space);
if (!colIndex) {
printf("ERROR: Could not allocate space for column indices: "FSTR64U"\n",
nent*sizeof(s64Int));
exit(EXIT_FAILURE);
}
#pragma omp parallel private (row, col, elm, first, last, iter)
{
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
printf("Number of threads     = %16d\n",nthread_input);
printf("Matrix order          = "FSTR64U"\n", size2);
printf("Stencil diameter      = %16d\n", 2*radius+1);
printf("Sparsity              = %16.10lf\n", sparsity);
printf("Number of iterations  = %16d\n", iterations);
#if SCRAMBLE
printf("Using scrambled indexing\n");
#else
printf("Using canonical indexing\n");
#endif
}
}
bail_out(num_error);
#pragma omp for
for (row=0; row<size2; row++) result[row] = vector[row] = 0.0;
#pragma omp for private (i,j,r)
for (row=0; row<size2; row++) {
j = row/size; i=row%size;
elm = row*stencil_size;
colIndex[elm] = REVERSE(LIN(i,j),lsize2);
for (r=1; r<=radius; r++, elm+=4) {
colIndex[elm+1] = REVERSE(LIN((i+r)%size,j),lsize2);
colIndex[elm+2] = REVERSE(LIN((i-r+size)%size,j),lsize2);
colIndex[elm+3] = REVERSE(LIN(i,(j+r)%size),lsize2);
colIndex[elm+4] = REVERSE(LIN(i,(j-r+size)%size),lsize2);
}
qsort(&(colIndex[row*stencil_size]), stencil_size, sizeof(s64Int), compare);
for (elm=row*stencil_size; elm<(row+1)*stencil_size; elm++)
matrix[elm] = 1.0/(double)(colIndex[elm]+1);
}
for (iter=0; iter<=iterations; iter++) {
if (iter == 1) {
#pragma omp barrier
#pragma omp master
{
sparse_time = wtime();
}
}
#pragma omp for
for (row=0; row<size2; row++) vector[row] += (double) (row+1);
#pragma omp for
for (row=0; row<size2; row++) {
first = stencil_size*row; last = first+stencil_size-1;
temp=0.0;
for (col=first; col<=last; col++) {
temp += matrix[col]*vector[colIndex[col]];
}
result[row] += temp;
}
} 
#pragma omp barrier
#pragma omp master
{
sparse_time = wtime() - sparse_time;
}
} 
reference_sum = 0.5 * (double) nent * (double) (iterations+1) *
(double) (iterations +2);
vector_sum = 0.0;
for (row=0; row<size2; row++) vector_sum += result[row];
if (ABS(vector_sum-reference_sum) > epsilon) {
printf("ERROR: Vector sum = %lf, Reference vector sum = %lf\n",
vector_sum, reference_sum);
exit(EXIT_FAILURE);
}
else {
printf("Solution validates\n");
#if VERBOSE
printf("Reference sum = %lf, vector sum = %lf\n",
reference_sum, vector_sum);
#endif
}
avgtime = sparse_time/iterations;
printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
1.0E-06 * (2.0*nent)/avgtime, avgtime);
exit(EXIT_SUCCESS);
}
u64Int reverse(register u64Int x, int shift_in_bits){
x = ((x >> 1)  & 0x5555555555555555) | ((x << 1)  & 0xaaaaaaaaaaaaaaaa);
x = ((x >> 2)  & 0x3333333333333333) | ((x << 2)  & 0xcccccccccccccccc);
x = ((x >> 4)  & 0x0f0f0f0f0f0f0f0f) | ((x << 4)  & 0xf0f0f0f0f0f0f0f0);
x = ((x >> 8)  & 0x00ff00ff00ff00ff) | ((x << 8)  & 0xff00ff00ff00ff00);
x = ((x >> 16) & 0x0000ffff0000ffff) | ((x << 16) & 0xffff0000ffff0000);
x = ((x >> 32) & 0x00000000ffffffff) | ((x << 32) & 0xffffffff00000000);
return (x>>((sizeof(u64Int)*BITS_IN_BYTE-shift_in_bits)));
}
int compare(const void *el1, const void *el2) {
s64Int v1 = *(s64Int *)el1;
s64Int v2 = *(s64Int *)el2;
return (v1<v2) ? -1 : (v1>v2) ? 1 : 0;
}
