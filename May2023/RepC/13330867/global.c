#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>
#define EOS '\0'
static int chartoi(char c) {
char letter[2]="0";
letter[0]=c;
return (atoi(letter));
}
int main(int argc, char ** argv)
{
int    my_ID;         
int    iterations;    
int    i, iter;       
int    checksum;      
char   *scramble = "27638472638746283742712311207892";
char   *basestring;   
char   *iterstring;   
char   *catstring;    
long   length;        
long   thread_length; 
int    basesum;       
double stopngo_time;  
int    nthread_input, 
nthread; 
int    num_error=0;   
printf("Parallel Research Kernels version %s\n", PRKVERSION);
printf("OpenMP global synchronization test\n");
if (argc != 4){
printf("Usage: %s <# threads> <# iterations> <scramble string length>\n", *argv);
exit(EXIT_FAILURE);
}
nthread_input = atoi(*++argv); 
if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
printf("ERROR: Invalid number of threads: %d\n", nthread_input);
exit(EXIT_FAILURE);
}
omp_set_num_threads(nthread_input);
iterations = atoi(*++argv);
if(iterations < 1){
printf("ERROR: iterations must be >= 1 : %d \n",iterations);
exit(EXIT_FAILURE);
}
length = atol(*++argv);
if (length <nthread_input || length%nthread_input !=0) {
printf("ERROR: length of string %ld must be multiple of # threads: %d\n", 
length, nthread_input);
exit(EXIT_FAILURE);
}
thread_length = length/nthread_input;
basestring = prk_malloc((thread_length+1)*sizeof(char));
if (basestring==NULL) {
printf("ERROR: Could not allocate space for scramble string\n");
exit(EXIT_FAILURE);
}
for (i=0; i<thread_length; i++) basestring[i]=scramble[i%32];
basestring[thread_length] = EOS;
catstring=(char *) prk_malloc((length+1)*sizeof(char));
if (catstring==NULL) {
printf("ERROR: Could not allocate space for concatenation string: %ld\n",
length+1);
exit(EXIT_FAILURE);
}
for (i=0; i<length; i++) catstring[i]='9';
catstring[length]=EOS;
#pragma omp parallel private(iterstring, my_ID, i, iter)
{
my_ID = omp_get_thread_num();
iterstring = (char *) prk_malloc((thread_length+1)*sizeof(char));
if (!iterstring) {
printf("ERROR: Thread %d could not allocate space for private string\n", 
my_ID);
num_error = 1;
}
bail_out(num_error);
strcpy(iterstring, basestring);
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
printf("Number of threads         = %d;\n",nthread_input);
printf("Length of scramble string = %ld\n", length);
printf("Number of iterations      = %d\n", iterations);
}
}
bail_out(num_error);
#pragma omp master 
{
stopngo_time = wtime();
}
for (iter=0; iter<iterations; iter++) { 
#pragma omp barrier
strncpy(catstring+my_ID*thread_length,iterstring,(size_t) thread_length);
#pragma omp barrier
for (i=0; i<thread_length; i++) iterstring[i]=catstring[my_ID+i*nthread];
#if VERBOSE
#pragma omp master
{
checksum=0;
for (i=0; i<strlen(catstring);i++) checksum+= chartoi(catstring[i]);
printf("Iteration %d, cat string %s, checksum equals: %d\n", 
iter, catstring, checksum);
}
#endif
}
#pragma omp master
{
stopngo_time = wtime() - stopngo_time;
}
} 
basesum=0;
for (i=0; i<thread_length; i++) basesum += chartoi(basestring[i]);
checksum=0;
{
size_t ist;
for (ist=0; ist<strlen(catstring);ist++) checksum += chartoi(catstring[ist]);
}
if (checksum != basesum*nthread) {
printf("Incorrect checksum: %d instead of %d\n", checksum, basesum*nthread);
exit(EXIT_FAILURE);
}
else {
#if VERBOSE
printf("Solution validates; Correct checksum of %d\n", checksum);
#else
printf("Solution validates\n");
#endif
}
printf("Rate (synch/s): %e, time (s): %lf\n", 
(((double)iterations)/stopngo_time), stopngo_time);
exit(EXIT_SUCCESS);
}  
