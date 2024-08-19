#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#if MKL
#include <mkl_cblas.h>
#endif
#define AA_arr(i,j) AA[(i)+(block+BOFFSET)*(j)]
#define BB_arr(i,j) BB[(i)+(block+BOFFSET)*(j)]
#define CC_arr(i,j) CC[(i)+(block+BOFFSET)*(j)]
#define  A_arr(i,j)  A[(i)+(order)*(j)]
#define  B_arr(i,j)  B[(i)+(order)*(j)]
#define  C_arr(i,j)  C[(i)+(order)*(j)]
#define forder (1.0*order)
int main(int argc, char **argv){
int     iter, i,ii,j,jj,k,kk,ig,jg,kg; 
int     iterations;           
double  dgemm_time,           
avgtime;
double  checksum = 0.0,       
ref_checksum;
double  epsilon = 1.e-8;      
int     nthread_input,        
nthread;   
int     num_error=0;          
static  
double  * RESTRICT A,         
* RESTRICT B,      
* RESTRICT C;
long    order;                
int     block;                
int     shortcut;             
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Dense matrix-matrix multiplication\n");
#if !MKL  
if (argc != 4 && argc != 5) {
printf("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\n",*argv);
#else
if (argc != 4) {
printf("Usage: %s <# threads> <# iterations> <matrix order>\n",*argv);
#endif
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
order = atol(*++argv);
if (order < 0) {
shortcut = 1;
order    = -order;
} else shortcut = 0;
if (order < 1) {
printf("ERROR: Matrix order must be positive: %ld\n", order);
exit(EXIT_FAILURE);
}
A = (double *) prk_malloc(order*order*sizeof(double));
B = (double *) prk_malloc(order*order*sizeof(double));
C = (double *) prk_malloc(order*order*sizeof(double));
if (!A || !B || !C) {
printf("ERROR: Could not allocate space for global matrices\n");
exit(EXIT_FAILURE);
}
ref_checksum = (0.25*forder*forder*forder*(forder-1.0)*(forder-1.0));
#pragma omp parallel for private(i,j) 
for(j = 0; j < order; j++) for(i = 0; i < order; i++) {
A_arr(i,j) = B_arr(i,j) = (double) j; 
C_arr(i,j) = 0.0;
}
#if !MKL
if (argc == 5) {
block = atoi(*++argv);
} else block = DEFAULTBLOCK;
#pragma omp parallel private (i,j,k,ii,jj,kk,ig,jg,kg,iter)
{
double * RESTRICT AA, * RESTRICT BB, * RESTRICT CC;
if (block > 0) {
AA = (double *) prk_malloc(block*(block+BOFFSET)*3*sizeof(double));
if (!AA) {
num_error = 1;
printf("Could not allocate space for matrix tiles on thread %d\n", 
omp_get_thread_num());
}
bail_out(num_error);
BB = AA + block*(block+BOFFSET);
CC = BB + block*(block+BOFFSET);
} 
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
printf("Matrix order          = %ld\n", order);
if (shortcut) 
printf("Only doing initialization\n"); 
printf("Number of threads     = %d\n", nthread_input);
if (block>0)
printf("Blocking factor       = %d\n", block);
else
printf("No blocking\n");
printf("Block offset          = %d\n", BOFFSET);
printf("Number of iterations  = %d\n", iterations);
printf("Using MKL library     = off\n");
}
}
bail_out(num_error); 
if (shortcut) exit(EXIT_SUCCESS);
for (iter=0; iter<=iterations; iter++) {
if (iter==1) {
#pragma omp barrier
#pragma omp master
{
dgemm_time = wtime();
}
}
if (block > 0) {
#pragma omp for 
for(jj = 0; jj < order; jj+=block){
for(kk = 0; kk < order; kk+=block) {
for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) 
for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++) 
BB_arr(j,k) =  B_arr(kg,jg);
for(ii = 0; ii < order; ii+=block){
for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
AA_arr(i,k) = A_arr(ig,kg);
for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) 
for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
CC_arr(i,j) = 0.0;
for (kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++)
for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) 
for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
CC_arr(i,j) += AA_arr(i,k)*BB_arr(j,k);
for (jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) 
for (ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++)
C_arr(ig,jg) += CC_arr(i,j);
}
}  
}
}
else {
#pragma omp for 
for (jg=0; jg<order; jg++) 
for (kg=0; kg<order; kg++) 
for (ig=0; ig<order; ig++) 
C_arr(ig,jg) += A_arr(ig,kg)*B_arr(kg,jg);
}
} 
#pragma omp barrier
#pragma omp master
{
dgemm_time = wtime() - dgemm_time;
}
} 
#else
printf("Matrix size           = %ldx%ld\n", order, order);
printf("Number of threads     = %d\n", nthread_input);
printf("Using MKL library     = on\n");
printf("Number of iterations  = %d\n", iterations);
for (iter=0; iter<=iterations; iter++) {
if (iter==1) dgemm_time = wtime();
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, order, order, 
order, 1.0, &(A_arr(0,0)), order, &(B_arr(0,0)), order, 
1.0, &(C_arr(0,0)), order);
}
dgemm_time = wtime()-dgemm_time;
#endif
for(checksum=0.0,j = 0; j < order; j++) for(i = 0; i < order; i++)
checksum += C_arr(i,j);
ref_checksum *= (iterations+1);
if (ABS((checksum - ref_checksum)/ref_checksum) > epsilon) {
printf("ERROR: Checksum = %lf, Reference checksum = %lf\n",
checksum, ref_checksum);
exit(EXIT_FAILURE);
}
else {
printf("Solution validates\n");
#if VERBOSE
printf("Reference checksum = %lf, checksum = %lf\n", 
ref_checksum, checksum);
#endif
}
double nflops = 2.0*forder*forder*forder;
avgtime = dgemm_time/iterations;
printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
1.0E-06 *nflops/avgtime, avgtime);
exit(EXIT_SUCCESS);
}
