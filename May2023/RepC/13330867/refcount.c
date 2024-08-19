#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define COUNTER1     (*pcounter1)
#define COUNTER2     (*pcounter2)
#define SCALAR       3.0
#define A0           0.0
#define B0           2.0
#define C0           2.0
#if INTEGER
#define DTYPE int64_t
#else
#define DTYPE double
#endif
void private_stream(double *a, double *b, double *c, size_t size) {
size_t j;
for (j=0; j<size; j++) a[j] += b[j] + SCALAR*c[j];
return;
}
#if LOCK==2 && _OPENMP>=201611
static omp_lock_hint_t parseLockHint (char const * hint)
{
static struct {char const * name; omp_lock_hint_t value;} keywords[] = {
{"none",        omp_lock_hint_none },
{"contended",   omp_lock_hint_contended},
{"uncontended", omp_lock_hint_uncontended},
{"speculative", omp_lock_hint_speculative}
};
if (!hint)
return omp_lock_hint_none;
int i;
for (i=0; i<sizeof(keywords)/sizeof(keywords[0]); i++) {
if (strcmp(keywords[i].name, hint) == 0)
return keywords[i].value;
}
printf ("***Unknown lock hint '%s'. Using 'none'***\n", hint);
return omp_lock_hint_none;
} 
#endif
int main(int argc, char ** argv)
{
size_t     iterations;      
size_t     stream_size;      
size_t     updates;         
int        page_fit;        
size_t     store_size;      
DTYPE      *pcounter1,     
*pcounter2;      
double     cosa, sina;      
DTYPE      *counter_space;  
#if UNUSED
DTYPE      refcounter1,
refcounter2;     
#endif
double     epsilon=1.e-7;   
omp_lock_t *pcounter_lock;  
double     refcount_time;   
int        nthread_input;   
int        nthread;         
#if _OPENMP>=201611
omp_lock_hint_t lock_hint;  
char const * lock_hint_name;
#endif
int        error=0;         
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP exclusive access test RefCount, shared counters\n");
#if LOCK==2 && _OPENMP>=201611
if (argc != 4 && argc != 5){
printf("Usage: %s <# threads> <# counter pair updates> <private stream size> [lock_hint]\n", *argv);
printf("    lock_hint is one of 'contended', 'uncontended', 'speculative', or 'none'\n");
return(1);
}
#else 
if (argc != 4){
printf("Usage: %s <# threads> <# counter pair updates> <private stream size>\n", *argv);
return(1);
}
#endif
nthread_input = atoi(*++argv);
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
iterations  = atol(*++argv);
if (iterations < 1){
printf("ERROR: iterations must be >= 1 : %zu \n",iterations);
exit(EXIT_FAILURE);
}
#if !STREAM
stream_size=0;
#else
stream_size = atol(*++argv);
if (stream_size < 0) {
printf("ERROR: private stream size %zu must be non-negative\n", stream_size);
exit(EXIT_FAILURE);
}
#endif
#if LOCK==2 && _OPENMP>=201611
lock_hint_name = (argc == 5) ? *++argv : "none"; 
lock_hint = parseLockHint(lock_hint_name);
#endif
omp_set_num_threads(nthread_input);
cosa = cos(1.0);
sina = sin(1.0);
#if !CONTENDED
#pragma omp parallel private(pcounter1,pcounter2,counter_space,pcounter_lock, page_fit,store_size)
#else
#pragma omp parallel 
#endif
{
size_t   iter, j;   
#if DEPENDENT
double tmp1;      
#endif
double *a, *b, *c;
int    num_error=0;
double aj, bj, cj;
DTYPE refcounter1, refcounter2;
if (stream_size) {
a = (double *) prk_malloc(3*sizeof(double)*stream_size);
if (!a) {
printf("ERROR: Could not allocate %ld words for private streams\n", 
3*sizeof(double)*stream_size);
exit(EXIT_FAILURE);
}
b = a + stream_size;
c = b + stream_size;
for (j=0; j<stream_size; j++) {
a[j] = A0;
b[j] = B0;
c[j] = C0;
}
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
printf("Number of threads              = %d\n",nthread_input);
printf("Number of counter pair updates = %ld\n", iterations);
#if STREAM
printf("Length of private stream       = %ld\n", stream_size);
#else
printf("Private stream disabled\n");
#endif
#if CONTENDED
printf("Counter access                 = contended\n");
#else
printf("Counter access                 = uncontended\n");
#endif
#if DEPENDENT
printf("Counter pair update type       = dependent\n");
#else
printf("Counter pair update type       = independent\n");
#endif
#if INTEGER
printf("Counter data type              = integer\n");
#else
printf("Counter data type              = floating point\n");
#endif
#if LOCK==2
printf("Mutex type                     = lock\n");
# if _OPENMP>=201611
printf("Lock hint                      = %s\n", lock_hint_name);
# endif
#elif LOCK==1
printf("Mutex type                     = atomic\n");
#else
printf("Mutex type                     = none\n");
#endif
}
}
bail_out(num_error);
#if CONTENDED
#pragma omp single
{
#endif
page_fit = 1;
store_size = sysconf(_SC_PAGESIZE);
#if VERBOSE
printf("Page size = %zu\n", store_size);
#endif
counter_space = (DTYPE *) prk_malloc(store_size+sizeof(DTYPE)+sizeof(omp_lock_t));
while (!counter_space && store_size>2*sizeof(DTYPE)) {
page_fit=0;
store_size/=2;
counter_space = (DTYPE *) prk_malloc(store_size+sizeof(DTYPE)+sizeof(omp_lock_t));
}
if (!counter_space) {
printf("ERROR: could not allocate space for counters\n");
exit(EXIT_FAILURE);
}
#if VERBOSE
if (!page_fit) printf("Counters do not fit on different pages\n");      
else           printf("Counters fit on different pages\n");      
#endif
pcounter1     = counter_space;
pcounter2     = counter_space + store_size/sizeof(DTYPE);
pcounter_lock = (omp_lock_t *)((char *)pcounter2+sizeof(DTYPE));
COUNTER1 = 1.0;
COUNTER2 = 0.0;
#if LOCK==2
#if _OPENMP>=201611
omp_init_lock_with_hint(pcounter_lock,lock_hint);
#else
omp_init_lock(pcounter_lock);
#endif
#endif
#if CONTENDED
}
#endif
#if DEPENDENT
#if LOCK==2
omp_set_lock(pcounter_lock);
#endif
tmp1 = COUNTER1;
COUNTER1 = cosa*tmp1 - sina*COUNTER2;
COUNTER2 = sina*tmp1 + cosa*COUNTER2;
#if LOCK==2
omp_unset_lock(pcounter_lock);
#endif
#else
#if LOCK==2
omp_set_lock(pcounter_lock);
COUNTER1++;
COUNTER2++;
omp_unset_lock(pcounter_lock);
#elif LOCK==1
#pragma omp atomic
COUNTER1++;
#pragma omp atomic
COUNTER2++;
#else
COUNTER1++;
COUNTER2++;
#endif
#endif
#if STREAM
private_stream(a, b, c, stream_size);
#endif
#pragma omp single
{
refcount_time = wtime();
}
#if CONTENDED 
#pragma omp for
for (iter=nthread; iter<=iterations; iter++) { 
#else
for (iter=1; iter<=iterations; iter++) { 
#endif
#if DEPENDENT
#if LOCK==2
omp_set_lock(pcounter_lock);
#endif
tmp1 = COUNTER1;
COUNTER1 = cosa*tmp1 - sina*COUNTER2;
COUNTER2 = sina*tmp1 + cosa*COUNTER2;
#if LOCK==2
omp_unset_lock(pcounter_lock);
#endif
#else
#if LOCK==2
omp_set_lock(pcounter_lock);
COUNTER1++;
COUNTER2++;
omp_unset_lock(pcounter_lock);
#elif LOCK==1
#pragma omp atomic
COUNTER1++;
#pragma omp atomic
COUNTER2++;
#else
COUNTER1++;
COUNTER2++;
#endif
#endif
#if STREAM
private_stream(a, b, c, stream_size);
#endif
}
#pragma omp single
{ 
refcount_time = wtime() - refcount_time;
}
aj = A0; bj = B0; cj = C0;
#if CONTENDED
#pragma omp for
#endif
for (iter=0; iter<=iterations; iter++) {
aj += bj + SCALAR*cj;
}
for (j=0; j<stream_size; j++) {
num_error += MAX(ABS(a[j]-aj)>epsilon,num_error);
}
if (num_error>0) {
printf("ERROR: Thread %d encountered errors in private work\n",
omp_get_thread_num());           
}
bail_out(num_error);
#if CONTENDED
#pragma omp master
{
#endif
#if DEPENDENT
refcounter1 = cos(iterations+1);
refcounter2 = sin(iterations+1);
#else
refcounter1 = (double)(iterations+2);
refcounter2 = (double)(iterations+1);
#endif
if ((ABS(COUNTER1-refcounter1)>epsilon) || 
(ABS(COUNTER2-refcounter2)>epsilon)) {
printf("ERROR: Incorrect or inconsistent counter values %13.10lf %13.10lf; ",
COUNTER1, COUNTER2);
printf("should be %13.10lf, %13.10lf\n", refcounter1, refcounter2);
num_error = 1;
}
#if !CONTENDED
#pragma omp critical
error = MAX(error,num_error);
#else
error = num_error;
#endif
#if CONTENDED
} 
#endif
} 
if (!error) {
#if VERBOSE
printf("Solution validates; Correct counter values %13.10lf %13.10lf\n", 
COUNTER1, COUNTER2);
#else
printf("Solution validates\n");
#endif
#if CONTENDED
updates=iterations-nthread;	
#else
updates=iterations*nthread;
#endif
printf("Rate (MCPUPs/s): %lf time (s): %lf\n", 
updates/refcount_time*1.e-6, refcount_time);
}
exit(EXIT_SUCCESS);
}
