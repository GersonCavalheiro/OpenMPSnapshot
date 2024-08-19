#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define ARRAY(i,j) vector[i+(j)*(m)]
#define LINEWORDS  16
#define flag(TID,j)    flag[((TID)+(j)*nthread)*LINEWORDS]
int main(int argc, char ** argv) {
int    TID;             
long   m, n;            
int    i, j, jj, iter, ID; 
int    iterations;      
int    *flag;           
int    *start, *end;    
int    segment_size;
double pipeline_time,   
avgtime; 
double epsilon = 1.e-8; 
double corner_val;      
int    nthread_input,   
nthread; 
int    grp;             
int    jjsize;          
double * RESTRICT vector;
long   total_length;    
int    num_error=0;     
int    true, false;     
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP pipeline execution on 2D grid\n");
if (argc != 5 && argc != 6){
printf("Usage: %s <# threads> <# iterations> <first array dimension> ", *argv);
printf("<second array dimension> [group factor]\n");
return(EXIT_FAILURE);
}
nthread_input = atoi(*++argv); 
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
iterations  = atoi(*++argv); 
if (iterations < 1){
printf("ERROR: iterations must be >= 1 : %d \n",iterations);
exit(EXIT_FAILURE);
}
m  = atol(*++argv);
n  = atol(*++argv);
if (m < 1 || n < 1){
printf("ERROR: grid dimensions must be positive: %ld, %ld \n", m, n);
exit(EXIT_FAILURE);
}
if (argc==6) {
grp = atoi(*++argv);
if (grp < 1) grp = 1;
else if (grp >= n) grp = n-1;
}
else grp = 1;
total_length = sizeof(double)*m*n;
vector = (double *) prk_malloc(total_length);
if (!vector) {
printf("ERROR: Could not allocate space for vector: %ld\n", total_length);
exit(EXIT_FAILURE);
}
if (m<nthread_input) {
printf("First grid dimension %ld smaller than number of threads requested: %d\n", 
m, nthread_input);
exit(EXIT_FAILURE);
}
start = (int *) prk_malloc(2*nthread_input*sizeof(int));
if (!start) {
printf("ERROR: Could not allocate space for array of slice boundaries\n");
exit(EXIT_FAILURE);
}
end = start + nthread_input;
start[0] = 0;
for (ID=0; ID<nthread_input; ID++) {
segment_size = m/nthread_input;
if (ID < (m%nthread_input)) segment_size++;
if (ID>0) start[ID] = end[ID-1]+1;
end[ID] = start[ID]+segment_size-1;
}
flag = (int *) prk_malloc(sizeof(int)*nthread_input*LINEWORDS*n);
if (!flag) {
printf("ERROR: COuld not allocate space for synchronization flags\n");
exit(EXIT_FAILURE);
}
#pragma omp parallel private(i, j, jj, jjsize, TID, iter, true, false) 
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
printf("Number of threads         = %d\n",nthread_input);
printf("Grid sizes                = %ld, %ld\n", m, n);
printf("Number of iterations      = %d\n", iterations);
if (grp > 1)
printf("Group factor              = %d (cheating!)\n", grp);
#if SYNCHRONOUS
printf("Neighbor thread handshake = on\n");
#else
printf("Neighbor thread handshake = off\n");
#endif
}
}
bail_out(num_error);
TID = omp_get_thread_num();
for (j=0; j<n; j++) for (i=start[TID]; i<=end[TID]; i++) ARRAY(i,j) = 0.0;
if (TID==0) for (j=0; j<n; j++) ARRAY(start[TID],j) = (double) j;
for (i=start[TID]; i<=end[TID]; i++) ARRAY(i,0) = (double) i;
true = 1; false = !true;
for (j=0; j<n; j++) flag(TID,j) = false;
#pragma omp barrier
for (iter = 0; iter<=iterations; iter++){
#if !SYNCHRONOUS
true = (iter+1)%2; false = !true;
#endif
if (iter == 1) { 
#pragma omp barrier
#pragma omp master
{
pipeline_time = wtime();
}
}
if (TID==0) { 
while (flag(0,0) == true) {
#pragma omp flush
}
#if SYNCHRONOUS
flag(0,0)= true;
#pragma omp flush
#endif      
}
for (j=1; j<n; j+=grp) { 
jjsize = MIN(grp, n-j);
if (TID > 0) {
while (flag(TID-1,j) == false) {
#pragma omp flush
}
#if SYNCHRONOUS
flag(TID-1,j)= false;
#pragma omp flush
#endif      
}
for (jj=j; jj<j+jjsize; jj++)
for (i=MAX(start[TID],1); i<= end[TID]; i++) {
ARRAY(i,jj) = ARRAY(i-1,jj) + ARRAY(i,jj-1) - ARRAY(i-1,jj-1);
}
if (TID < nthread-1) {
#if SYNCHRONOUS 
while (flag(TID,j) == true) {
#pragma omp flush
}
#endif 
flag(TID,j) = true;
#pragma omp flush
}
}
if (TID==nthread-1) { 
ARRAY(0,0) = -ARRAY(m-1,n-1);
#if SYNCHRONOUS
while (flag(0,0) == false) {
#pragma omp flush
}
flag(0,0) = false;
#else
#pragma omp flush
flag(0,0) = true;
#endif
#pragma omp flush
}
} 
#pragma omp barrier
#pragma omp master
{
pipeline_time = wtime() - pipeline_time;
}
} 
corner_val = (double)((iterations+1)*(n+m-2));
if (fabs(ARRAY(m-1,n-1)-corner_val)/corner_val > epsilon) {
printf("ERROR: checksum %lf does not match verification value %lf\n",
ARRAY(m-1,n-1), corner_val);
exit(EXIT_FAILURE);
}
#if VERBOSE   
printf("Solution validates; verification value = %lf\n", corner_val);
printf("Point-to-point synchronizations/s: %lf\n",
((float)((n-1)*(nthread-1)))/(avgtime));
#else
printf("Solution validates\n");
#endif
avgtime = pipeline_time/iterations;
if (grp>1) avgtime *= -1.0;
printf("Rate (MFlops/s): %lf Avg time (s): %lf\n",
1.0E-06 * 2 * ((double)((m-1)*(n-1)))/avgtime, avgtime);
exit(EXIT_SUCCESS);
}
