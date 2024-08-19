

#include "omp.h"
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

#define T_BENCHMARKING (0)
#define T_INITIALIZATION (1)
#define T_SORTING (2)
#define T_TOTAL_EXECUTION (3)












#define USE_BUCKETS




#ifndef CLASS
#define CLASS 'S'
#endif




#if CLASS == 'S'
#define TOTAL_KEYS_LOG_2 16
#define MAX_KEY_LOG_2 11
#define NUM_BUCKETS_LOG_2 9
#endif




#if CLASS == 'W'
#define TOTAL_KEYS_LOG_2 20
#define MAX_KEY_LOG_2 16
#define NUM_BUCKETS_LOG_2 10
#endif




#if CLASS == 'A'
#define TOTAL_KEYS_LOG_2 23
#define MAX_KEY_LOG_2 19
#define NUM_BUCKETS_LOG_2 10
#endif




#if CLASS == 'B'
#define TOTAL_KEYS_LOG_2 25
#define MAX_KEY_LOG_2 21
#define NUM_BUCKETS_LOG_2 10
#endif




#if CLASS == 'C'
#define TOTAL_KEYS_LOG_2 27
#define MAX_KEY_LOG_2 23
#define NUM_BUCKETS_LOG_2 10
#endif




#if CLASS == 'D'
#define TOTAL_KEYS_LOG_2 31
#define MAX_KEY_LOG_2 27
#define NUM_BUCKETS_LOG_2 10
#endif

#if CLASS == 'D'
#define TOTAL_KEYS (1L << TOTAL_KEYS_LOG_2)
#else
#define TOTAL_KEYS (1 << TOTAL_KEYS_LOG_2)
#endif
#define MAX_KEY (1 << MAX_KEY_LOG_2)
#define NUM_BUCKETS (1 << NUM_BUCKETS_LOG_2)
#define NUM_KEYS TOTAL_KEYS
#define SIZE_OF_BUFFERS NUM_KEYS                                           

#define MAX_ITERATIONS 10
#define TEST_ARRAY_SIZE 5






#if CLASS == 'D'
typedef long INT_TYPE;
#else
typedef int INT_TYPE;
#endif




INT_TYPE* key_buff_ptr_global; 
int passed_verification;                                 





#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
INT_TYPE key_array[SIZE_OF_BUFFERS];
INT_TYPE key_buff1[MAX_KEY];
INT_TYPE key_buff2[SIZE_OF_BUFFERS];
INT_TYPE partial_verify_vals[TEST_ARRAY_SIZE];
INT_TYPE** key_buff1_aptr = NULL;
#else
INT_TYPE (*key_array)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(SIZE_OF_BUFFERS));    
INT_TYPE (*key_buff1)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(MAX_KEY));                
INT_TYPE (*key_buff2)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(SIZE_OF_BUFFERS));
INT_TYPE (*partial_verify_vals)=(INT_TYPE*)malloc(sizeof(INT_TYPE)*(TEST_ARRAY_SIZE));       
INT_TYPE** key_buff1_aptr = NULL;
#endif

#ifdef USE_BUCKETS
INT_TYPE** bucket_size; 
INT_TYPE bucket_ptrs[NUM_BUCKETS];
#pragma omp threadprivate(bucket_ptrs)
#endif




INT_TYPE test_index_array[TEST_ARRAY_SIZE],
test_rank_array[TEST_ARRAY_SIZE],

S_test_index_array[TEST_ARRAY_SIZE] = 
{48427,17148,23627,62548,4431},
S_test_rank_array[TEST_ARRAY_SIZE] = 
{0,18,346,64917,65463},

W_test_index_array[TEST_ARRAY_SIZE] = 
{357773,934767,875723,898999,404505},
W_test_rank_array[TEST_ARRAY_SIZE] = 
{1249,11698,1039987,1043896,1048018},

A_test_index_array[TEST_ARRAY_SIZE] = 
{2112377,662041,5336171,3642833,4250760},
A_test_rank_array[TEST_ARRAY_SIZE] = 
{104,17523,123928,8288932,8388264},

B_test_index_array[TEST_ARRAY_SIZE] = 
{41869,812306,5102857,18232239,26860214},
B_test_rank_array[TEST_ARRAY_SIZE] = 
{33422937,10244,59149,33135281,99}, 

C_test_index_array[TEST_ARRAY_SIZE] = 
{44172927,72999161,74326391,129606274,21736814},
C_test_rank_array[TEST_ARRAY_SIZE] = 
{61147,882988,266290,133997595,133525895},

D_test_index_array[TEST_ARRAY_SIZE] = 
{1317351170,995930646,1157283250,1503301535,1453734525},
D_test_rank_array[TEST_ARRAY_SIZE] = 
{1,36538729,1978098519,2145192618,2147425337};




void alloc_key_buff();
void* alloc_mem(size_t size);
void create_seq(double seed,
double a);
double find_my_seed(int kn,
int np,
long nn,
double s,
double a );
void full_verify();    
void rank(int iteration);




int main(int argc, char** argv){
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif
int i, iteration, timer_on;
double timecounter;
FILE* fp;


timer_on = 0;            
if((fp = fopen("timer.flag", "r")) != NULL){
fclose(fp);
timer_on = 1;
}
timer_clear( T_BENCHMARKING );
if(timer_on){
timer_clear( T_INITIALIZATION );
timer_clear( T_SORTING );
timer_clear( T_TOTAL_EXECUTION );
}

if(timer_on)timer_start( T_TOTAL_EXECUTION );


for( i=0; i<TEST_ARRAY_SIZE; i++ )
switch( CLASS )
{
case 'S':
test_index_array[i] = S_test_index_array[i];
test_rank_array[i]  = S_test_rank_array[i];
break;
case 'A':
test_index_array[i] = A_test_index_array[i];
test_rank_array[i]  = A_test_rank_array[i];
break;
case 'W':
test_index_array[i] = W_test_index_array[i];
test_rank_array[i]  = W_test_rank_array[i];
break;
case 'B':
test_index_array[i] = B_test_index_array[i];
test_rank_array[i]  = B_test_rank_array[i];
break;
case 'C':
test_index_array[i] = C_test_index_array[i];
test_rank_array[i]  = C_test_rank_array[i];
break;
case 'D':
test_index_array[i] = D_test_index_array[i];
test_rank_array[i]  = D_test_rank_array[i];
break;
};        


printf("\n\n NAS Parallel Benchmarks 4.1 Parallel C++ version with OpenMP - IS Benchmark\n\n");
printf(" Size:  %ld  (class %c)\n", (long)TOTAL_KEYS, CLASS);
printf(" Iterations:   %d\n", MAX_ITERATIONS);
printf( "\n" );

if(timer_on)timer_start( T_INITIALIZATION );


create_seq(314159265.00 , 
1220703125.00 );                 

alloc_key_buff();
if(timer_on)timer_stop( T_INITIALIZATION );



rank( 1 );  


passed_verification = 0;

if( CLASS != 'S' ) printf( "\n   iteration\n" );


timer_start( T_BENCHMARKING );


for(iteration=1; iteration<=MAX_ITERATIONS; iteration++){
if(CLASS != 'S')printf("        %d\n", iteration);
rank( iteration );
}


timer_stop( T_BENCHMARKING );
timecounter = timer_read( T_BENCHMARKING );



if(timer_on)timer_start( T_SORTING );
full_verify();
if(timer_on)timer_stop( T_SORTING );

if(timer_on)timer_stop( T_TOTAL_EXECUTION );    


if(passed_verification != 5*MAX_ITERATIONS + 1){passed_verification = 0;}
setenv("OMP_NUM_THREADS","1",0);
c_print_results((char*)"IS",
CLASS,
(int)(TOTAL_KEYS/64),
64,
0,
MAX_ITERATIONS,
timecounter,
((double)(MAX_ITERATIONS*TOTAL_KEYS))/timecounter/1000000.0,
(char*)"keys ranked",
passed_verification,
(char*)NPBVERSION,
(char*)COMPILETIME,
(char*)COMPILERVERSION,
(char*)LIBVERSION,
std::getenv("OMP_NUM_THREADS"),
(char*)CS1,
(char*)CS2,
(char*)CS3,
(char*)CS4,
(char*)CS5,
(char*)CS6,
(char*)CS7);


if(timer_on){
double t_total, t_percent;
t_total = timer_read( T_TOTAL_EXECUTION );
printf("\nAdditional timers -\n");
printf(" Total execution: %8.3f\n", t_total);
if (t_total == 0.0) t_total = 1.0;
timecounter = timer_read(T_INITIALIZATION);
t_percent = timecounter/t_total * 100.;
printf(" Initialization : %8.3f (%5.2f%%)\n", timecounter, t_percent);
timecounter = timer_read(T_BENCHMARKING);
t_percent = timecounter/t_total * 100.;
printf(" Benchmarking   : %8.3f (%5.2f%%)\n", timecounter, t_percent);
timecounter = timer_read(T_SORTING);
t_percent = timecounter/t_total * 100.;
printf(" Sorting        : %8.3f (%5.2f%%)\n", timecounter, t_percent);
}

return 0;
}

void alloc_key_buff(){
INT_TYPE i;
int num_procs;

num_procs = omp_get_max_threads();

#ifdef USE_BUCKETS
bucket_size = (INT_TYPE**)alloc_mem(sizeof(INT_TYPE*)*num_procs);

for(i = 0; i < num_procs; i++){
bucket_size[i] = (INT_TYPE*)alloc_mem(sizeof(INT_TYPE)*NUM_BUCKETS);
}

#pragma omp parallel for
for( i=0; i<NUM_KEYS; i++ )
key_buff2[i] = 0;
#else 
key_buff1_aptr = (INT_TYPE**)alloc_mem(sizeof(INT_TYPE*)*num_procs);

key_buff1_aptr[0] = key_buff1;
for(i = 1; i < num_procs; i++) {
key_buff1_aptr[i] = (INT_TYPE *)alloc_mem(sizeof(INT_TYPE) * MAX_KEY);
}
#endif 
}




void* alloc_mem(size_t size){
void* p;
p = (void*)malloc(size);
if(!p){
perror("Memory allocation error");
exit(1);
}
return p;
}




void create_seq(double seed,
double a){
#pragma omp parallel
{
double x, s;
INT_TYPE i, k;

INT_TYPE k1, k2;
double an = a;
int myid, num_procs;
INT_TYPE mq;

myid = omp_get_thread_num();
num_procs = omp_get_num_threads();

mq = (NUM_KEYS + num_procs - 1) / num_procs;
k1 = mq * myid;
k2 = k1 + mq;
if ( k2 > NUM_KEYS ) k2 = NUM_KEYS;

s = find_my_seed( myid, 
num_procs,
(long)4*NUM_KEYS,
seed,
an );

k = MAX_KEY/4;

for(i=k1; i<k2; i++){
x = randlc(&s, an);
x += randlc(&s, an);
x += randlc(&s, an);
x += randlc(&s, an);
key_array[i] = k*x;
}
} 
}






double find_my_seed(int kn, 
int np, 
long nn, 
double s, 
double a){ 

double t1,t2;
long mq,nq,kk,ik;

if ( kn == 0 ) return s;

mq = (nn/4 + np - 1) / np;
nq = mq * 4 * kn; 

t1 = s;
t2 = a;
kk = nq;
while( kk > 1 ){
ik = kk / 2;
if(2 * ik ==  kk){
(void)randlc( &t2, t2 );
kk = ik;
}
else{
(void)randlc( &t1, t2 );
kk = kk - 1;
}
}
(void)randlc( &t1, t2 );

return( t1 );
}




void full_verify(){
INT_TYPE i, j;
INT_TYPE k, k1, k2;
int myid, num_procs;

myid = 0;
num_procs = 1;




#ifdef USE_BUCKETS

#pragma omp parallel for private(i,j,k,k1) schedule(dynamic)
for( j=0; j< NUM_BUCKETS; j++ ) {
k1 = (j > 0)? bucket_ptrs[j-1] : 0;
for ( i = k1; i < bucket_ptrs[j]; i++ ) {
k = --key_buff_ptr_global[key_buff2[i]];
key_array[k] = key_buff2[i];
}
}
#else    
for( i=0; i<NUM_KEYS; i++ )
key_buff2[i] = key_array[i];

j = num_procs;
j = (MAX_KEY + j - 1) / j;
k1 = j * myid;
k2 = k1 + j;
if (k2 > MAX_KEY) k2 = MAX_KEY;
for( i=0; i<NUM_KEYS; i++ ) {
if (key_buff2[i] >= k1 && key_buff2[i] < k2) {
k = --key_buff_ptr_global[key_buff2[i]];
key_array[k] = key_buff2[i];
}
}
#endif


j = 0;
#pragma omp parallel for reduction(+:j)
for( i=1; i<NUM_KEYS; i++ )
if( key_array[i-1] > key_array[i] )
j++;
if( j != 0 )
printf( "Full_verify: number of keys out of sort: %ld\n", (long)j );
else
passed_verification++;
}




void rank(int iteration){

INT_TYPE i, k;
INT_TYPE *key_buff_ptr, *key_buff_ptr2;

#ifdef USE_BUCKETS
int shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
INT_TYPE num_bucket_keys = (1L << shift);
#endif

key_array[iteration] = iteration;
key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;



for( i=0; i<TEST_ARRAY_SIZE; i++ )
partial_verify_vals[i] = key_array[test_index_array[i]];


#ifdef USE_BUCKETS
key_buff_ptr2 = key_buff2;
#else
key_buff_ptr2 = key_array;
#endif
key_buff_ptr = key_buff1;





#ifdef USE_BUCKETS
#pragma omp parallel private(i, k)
{
INT_TYPE *work_buff, m, k1, k2;

int myid = omp_get_thread_num();
int num_procs = omp_get_num_threads();

work_buff = bucket_size[myid];


for( i=0; i<NUM_BUCKETS; i++ )  
work_buff[i] = 0;


#pragma omp for schedule(static)
for( i=0; i<NUM_KEYS; i++ )
work_buff[key_array[i] >> shift]++;



bucket_ptrs[0] = 0;
for( k=0; k< myid; k++ )  
bucket_ptrs[0] += bucket_size[k][0];

for( i=1; i< NUM_BUCKETS; i++ ) { 
bucket_ptrs[i] = bucket_ptrs[i-1];
for( k=0; k< myid; k++ )
bucket_ptrs[i] += bucket_size[k][i];
for( k=myid; k< num_procs; k++ )
bucket_ptrs[i] += bucket_size[k][i-1];
}


#pragma omp for schedule(static)
for( i=0; i<NUM_KEYS; i++ ){
k = key_array[i];
key_buff2[bucket_ptrs[k >> shift]++] = k;
}


if (myid < num_procs-1) {
for( i=0; i< NUM_BUCKETS; i++ )
for( k=myid+1; k< num_procs; k++ )
bucket_ptrs[i] += bucket_size[k][i];
}





#pragma omp for schedule(dynamic)
for( i=0; i< NUM_BUCKETS; i++ ) {

k1 = i * num_bucket_keys;
k2 = k1 + num_bucket_keys;
for ( k = k1; k < k2; k++ )
key_buff_ptr[k] = 0;




m = (i > 0)? bucket_ptrs[i-1] : 0;
for ( k = m; k < bucket_ptrs[i]; k++ )
key_buff_ptr[key_buff_ptr2[k]]++; 



key_buff_ptr[k1] += m;
for ( k = k1+1; k < k2; k++ )
key_buff_ptr[k] += key_buff_ptr[k-1];
}
}
#else 
work_buff = key_buff1_aptr[myid];

for( i=0; i<MAX_KEY; i++ )
work_buff[i] = 0;




for( i=0; i<NUM_KEYS; i++ )
work_buff[key_buff_ptr2[i]]++; 

for( i=0; i<MAX_KEY-1; i++ )   
work_buff[i+1] += work_buff[i];

for( k=1; k<num_procs; k++ ){
for( i=0; i<MAX_KEY; i++ )
key_buff_ptr[i] += key_buff1_aptr[k][i];
}
#endif   




for( i=0; i<TEST_ARRAY_SIZE; i++ ){                                             
k = partial_verify_vals[i]; 
if( 0 < k  &&  k <= NUM_KEYS-1 )
{
INT_TYPE key_rank = key_buff_ptr[k-1];
int failed = 0;

switch( CLASS )
{
case 'S':
if( i <= 2 )
{
if( key_rank != test_rank_array[i]+iteration )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-iteration )
failed = 1;
else
passed_verification++;
}
break;
case 'W':
if( i < 2 )
{
if( key_rank != test_rank_array[i]+(iteration-2) )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-iteration )
failed = 1;
else
passed_verification++;
}
break;
case 'A':
if( i <= 2 )
{
if( key_rank != test_rank_array[i]+(iteration-1) )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-(iteration-1) )
failed = 1;
else
passed_verification++;
}
break;
case 'B':
if( i == 1 || i == 2 || i == 4 )
{
if( key_rank != test_rank_array[i]+iteration )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-iteration )
failed = 1;
else
passed_verification++;
}
break;
case 'C':
if( i <= 2 )
{
if( key_rank != test_rank_array[i]+iteration )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-iteration )
failed = 1;
else
passed_verification++;
}
break;
case 'D':
if( i < 2 )
{
if( key_rank != test_rank_array[i]+iteration )
failed = 1;
else
passed_verification++;
}
else
{
if( key_rank != test_rank_array[i]-iteration )
failed = 1;
else
passed_verification++;
}
break;
}
if( failed == 1 )
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, (int)i );
}
}




if( iteration == MAX_ITERATIONS ) 
key_buff_ptr_global = key_buff_ptr;
}
