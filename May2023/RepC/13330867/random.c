#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#if LONG_IS_64BITS 
#define POLY               0x0000000000000007UL
#define PERIOD             1317624576693539401L
#define SEQSEED            834568137686317453L
#else 
#define POLY               0x0000000000000007ULL
#define PERIOD             1317624576693539401LL
#define SEQSEED            834568137686317453LL
#endif 
#if HPCC
#undef  ATOMIC
#undef  CHUNKED
#undef  ERRORPERCENT
#define ERRORPERCENT 1
#else
#if CHUNKED
#undef ATOMIC
#endif
#endif
static u64Int PRK_starts(s64Int);
#if UNUSED
static int    poweroftwo(int);
#endif
int main(int argc, char **argv) {
int               my_ID;       
int               update_ratio;  
int               nstarts;     
s64Int            i, j, round, oldsize; 
s64Int            error;       
s64Int            tablesize;   
s64Int            nupdate;     
size_t            tablespace;  
u64Int            *ran;        
s64Int            index;       
#if VERBOSE
u64Int * RESTRICT Hist;        
unsigned int      *HistHist;   
#endif
u64Int * RESTRICT Table;       
double            random_time;
int               nthread_input;   
int               nthread;
int               log2tablesize; 
int               num_error=0; 
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP Random Access test\n");
#if LONG_IS_64BITS
if (sizeof(long) != 8) {
printf("ERROR: Makefile says \"long\" is 8 bytes, but it is %d bytes\n",
sizeof(long)); 
exit(EXIT_FAILURE);
}
#endif
if (argc != 5){
printf("Usage: %s <# threads> <log2 tablesize> <#update ratio> ", *argv);
printf("<vector length>\n");
exit(EXIT_FAILURE);
}
nthread_input = atoi(*++argv);
if (nthread_input <1) {
printf("ERROR: Invalid number of threads: %d, must be positive\n",
nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
log2tablesize  = atoi(*++argv);
if (log2tablesize < 1){
printf("ERROR: Log2 tablesize is %d; must be >= 1\n",log2tablesize);
exit(EXIT_FAILURE);
}
update_ratio  = atoi(*++argv);
if (update_ratio <1) {
printf("ERROR: Invalid update ratio: %d, must be positive\n", 
update_ratio);
exit(EXIT_FAILURE);
}
nstarts = atoi(*++argv);
if (nstarts <1) {
printf("ERROR: Invalid vector length: %d, must be positive\n",
nstarts);
exit(EXIT_FAILURE);
}
if (nstarts%nthread_input) {
printf("ERROR: vector length %d must be divisible by # threads %d\n",
nstarts, nthread_input);
exit(EXIT_FAILURE);
}
if (update_ratio%nstarts) {
printf("ERROR: update ratio %d must be divisible by vector length %d\n",
update_ratio, nstarts);
exit(EXIT_FAILURE);
}
tablesize = 1;
for (i=0; i<log2tablesize; i++) {
oldsize =  tablesize;
tablesize <<=1;
if (tablesize/2 != oldsize) {
printf("Requested table size too large; reduce log2 tablesize = %d\n",
log2tablesize);
exit(EXIT_FAILURE);
}
}
tablespace = (size_t) tablesize*sizeof(u64Int);
if ((tablespace/sizeof(u64Int)) != tablesize || tablespace <=0) {
printf("Cannot represent space for table on this system; ");
printf("reduce log2 tablesize\n");
exit(EXIT_FAILURE);
}
#if VERBOSE
Hist = (u64Int *) prk_malloc(tablespace);
HistHist = (unsigned int *) prk_malloc(tablespace);
if (!Hist || ! HistHist) {
printf("ERROR: Could not allocate space for histograms\n");
exit(EXIT_FAILURE);
}
#endif
nupdate = update_ratio * tablesize;
if (nupdate/tablesize != update_ratio) {
printf("Requested number of updates too large; ");
printf("reduce log2 tablesize or update ratio\n");
exit(EXIT_FAILURE);
}
Table = (u64Int *) prk_malloc(tablespace);
if (!Table) {
printf("ERROR: Could not allocate space of "FSTR64U"  bytes for table\n",
(u64Int) tablespace);
exit(EXIT_FAILURE);
}
error = 0;
#pragma omp parallel private(i, j, ran, round, index, my_ID) reduction(+:error)
{
int my_starts;
my_ID   = omp_get_thread_num();
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
printf("Number of threads      = "FSTR64U"\n", (u64Int) nthread_input);
printf("Table size (shared)    = "FSTR64U"\n", tablesize);
printf("Update ratio           = "FSTR64U"\n", (u64Int) update_ratio);
printf("Number of updates      = "FSTR64U"\n", nupdate);
printf("Vector length          = "FSTR64U"\n", (u64Int) nstarts);
printf("Percent errors allowed = "FSTR64U"\n", (u64Int) ERRORPERCENT);
#if RESTRICT_KEYWORD
printf("No aliasing            = on\n");
#else
printf("No aliasing            = off\n");
#endif
#if defined(ATOMIC) && !defined(CHUNKED)
printf("Shared table, atomic updates\n");
#elif defined(CHUNKED)
printf("Shared, chunked table\n");
#else
printf("Shared table, non-atomic updates\n");
#endif
}
}
bail_out(num_error);
#if CHUNKED
u64Int low =  my_ID   *(tablesize/nthread);
u64Int up  = (my_ID+1)*(tablesize/nthread);
my_starts = nstarts;
#else
my_starts = nstarts/nthread;
#endif
ran = (u64Int *) prk_malloc(my_starts*sizeof(u64Int));
if (!ran) {
printf("ERROR: Thread %d Could not allocate %d bytes for random numbers\n",
my_ID, my_starts*(int)sizeof(u64Int));
num_error = 1;
}
bail_out(num_error);
#pragma omp for 
for(i=0;i<tablesize;i++) Table[i] = (u64Int) i;
#pragma omp barrier
#pragma omp master
{
random_time = wtime();
}
#if CHUNKED
int offset = 0;
#else
int offset = my_ID*my_starts;
#endif
for (round=0; round <2; round++) {
for (j=0; j<my_starts; j++) {
ran[j] = PRK_starts(SEQSEED+(nupdate/nstarts)*(j+offset));
}
for (j=0; j<my_starts; j++) {
for (i=0; i<nupdate/(nstarts*2); i++) {
ran[j] = (ran[j] << 1) ^ ((s64Int)ran[j] < 0? POLY: 0);
index = ran[j] & (tablesize-1);
#if defined(ATOMIC) 
#pragma omp atomic      
#elif defined(CHUNKED)
if (index >= low && index < up) {
#endif
Table[index] ^= ran[j];
#if VERBOSE
#pragma omp atomic
Hist[index] += 1;
#endif
#if CHUNKED
}
#endif
}
}
}
#pragma omp master 
{ 
random_time = wtime() - random_time;
}
} 
for(i=0;i<tablesize;i++) {
if(Table[i] != (u64Int) i) {
#if VERBOSE
printf("Error Table["FSTR64U"]="FSTR64U"\n",i,Table[i]);
#endif
error++;
}
}
if ((error && (ERRORPERCENT==0)) ||
((double)error/(double)tablesize > ((double) ERRORPERCENT)*0.01)) {
printf("ERROR: number of incorrect table elements = "FSTR64U"\n", error);
exit(EXIT_FAILURE);
}
else {
printf("Solution validates, number of errors: %ld\n",(long) error);
printf("Rate (GUPs/s): %lf, time (s) = %lf\n", 
1.e-9*nupdate/random_time,random_time);
}
#if VERBOSE
for(i=0;i<tablesize;i++) HistHist[Hist[i]]+=1;
for(i=0;i<=tablesize;i++) if (HistHist[i] != 0)
printf("HistHist[%4.1d]=%9.1d\n",(int)i,HistHist[i]);
#endif
exit(EXIT_SUCCESS);
}
u64Int PRK_starts(s64Int n)
{ 
int i, j; 
u64Int m2[64];
u64Int temp, ran; 
while (n < 0) n += PERIOD;
while (n > PERIOD) n -= PERIOD;
if (n == 0) return 0x1;
temp = 0x1;
for (i=0; i<64; i++) {
m2[i] = temp;
temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0); 
temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0); 
} 
for (i=62; i>=0; i--) 
if ((n >> i) & 1) 
break; 
ran = 0x2;    
while (i > 0) { 
temp = 0; 
for (j=0; j<64; j++)
if ((unsigned int)((ran >> j) & 1)) 
temp ^= m2[j]; 
ran = temp; 
i -= 1; 
if ((n >> i) & 1)
ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0); 
} 
return ran; 
} 
#if UNUSED
int poweroftwo(int n) {
int log2n = 0;
while ((1<<log2n)<n) log2n++;
if (1<<log2n != n) return (-1);
else               return (log2n);
}
#endif
