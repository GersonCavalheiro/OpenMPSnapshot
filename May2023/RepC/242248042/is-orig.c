#include "npbparams.h"
#include <stdlib.h>
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#endif 
extern void timer_clear(int);
extern void timer_start(int);
extern void timer_stop(int);
extern double timer_read(int);
extern void c_print_results(char *name, char class, int n1, int n2,
int n3, int niter, int nthreads, double t,
double mops, char *optype, int passed_verification,
char *npbversion, char *compiletime, char *cc,
char *clink, char *c_lib, char *c_inc,
char *cflags, char *clinkflags, char *rand);
#ifndef CLASS
#define CLASS 'S'
#endif
#if CLASS == 'S'
#define  TOTAL_KEYS_LOG_2    16
#define  MAX_KEY_LOG_2       11
#define  NUM_BUCKETS_LOG_2   9
#endif
#if CLASS == 'W'
#define  TOTAL_KEYS_LOG_2    20
#define  MAX_KEY_LOG_2       16
#define  NUM_BUCKETS_LOG_2   10
#endif
#if CLASS == 'A'
#define  TOTAL_KEYS_LOG_2    23
#define  MAX_KEY_LOG_2       19
#define  NUM_BUCKETS_LOG_2   10
#endif
#if CLASS == 'B'
#define  TOTAL_KEYS_LOG_2    25
#define  MAX_KEY_LOG_2       21
#define  NUM_BUCKETS_LOG_2   10
#endif
#if CLASS == 'C'
#define  TOTAL_KEYS_LOG_2    27
#define  MAX_KEY_LOG_2       23
#define  NUM_BUCKETS_LOG_2   10
#endif
#define  TOTAL_KEYS          (1 << TOTAL_KEYS_LOG_2)
#define  MAX_KEY             (1 << MAX_KEY_LOG_2)
#define  NUM_BUCKETS         (1 << NUM_BUCKETS_LOG_2)
#define  NUM_KEYS            TOTAL_KEYS
#define  SIZE_OF_BUFFERS     NUM_KEYS  
#define  MAX_ITERATIONS      10
#define  TEST_ARRAY_SIZE     5
typedef  int  INT_TYPE;
INT_TYPE *key_buff_ptr_global;         
int      passed_verification;
INT_TYPE key_array[SIZE_OF_BUFFERS],    
key_buff1[SIZE_OF_BUFFERS],    
key_buff2[SIZE_OF_BUFFERS],
partial_verify_vals[TEST_ARRAY_SIZE];
#ifdef USE_BUCKETS
INT_TYPE bucket_size[NUM_BUCKETS],                    
bucket_ptrs[NUM_BUCKETS];
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
{61147,882988,266290,133997595,133525895};
double	randlc( double *X, double *A );
void full_verify( void );
double	randlc(X, A)
double *X;
double *A;
{
static int        KS=0;
static double	R23, R46, T23, T46;
double		T1, T2, T3, T4;
double		A1;
double		A2;
double		X1;
double		X2;
double		Z;
int     		i, j;
if (KS == 0) 
{
R23 = 1.0;
R46 = 1.0;
T23 = 1.0;
T46 = 1.0;
for (i=1; i<=23; i++)
{
R23 = 0.50 * R23;
T23 = 2.0 * T23;
}
for (i=1; i<=46; i++)
{
R46 = 0.50 * R46;
T46 = 2.0 * T46;
}
KS = 1;
}
T1 = R23 * *A;
j  = T1;
A1 = j;
A2 = *A - T23 * A1;
T1 = R23 * *X;
j  = T1;
X1 = j;
X2 = *X - T23 * X1;
T1 = A1 * X2 + A2 * X1;
j  = R23 * T1;
T2 = j;
Z = T1 - T23 * T2;
T3 = T23 * Z + A2 * X2;
j  = R46 * T3;
T4 = j;
*X = T3 - T46 * T4;
return(R46 * *X);
} 
void	create_seq( double seed, double a )
{
double x;
int    i, j, k;
k = MAX_KEY/4;
for (i=0; i<NUM_KEYS; i++)
{
x = randlc(&seed, &a);
x += randlc(&seed, &a);
x += randlc(&seed, &a);
x += randlc(&seed, &a);  
key_array[i] = k*x;
}
}
void full_verify()
{
INT_TYPE    i, j;
INT_TYPE    k;
INT_TYPE    m, unique_keys;
for( i=0; i<NUM_KEYS; i++ )
key_array[--key_buff_ptr_global[key_buff2[i]]] = key_buff2[i];
j = 0;
for( i=1; i<NUM_KEYS; i++ )
if( key_array[i-1] > key_array[i] )
j++;
if( j != 0 )
{
printf( "Full_verify: number of keys out of sort: %d\n",
j );
}
else
passed_verification++;
}
void rank( int iteration )
{
INT_TYPE    i, j, k;
INT_TYPE    l, m;
INT_TYPE    shift = MAX_KEY_LOG_2 - NUM_BUCKETS_LOG_2;
INT_TYPE    key;
INT_TYPE    min_key_val, max_key_val;
INT_TYPE	prv_buff1[MAX_KEY];
#pragma omp master
{
key_array[iteration] = iteration;
key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
for( i=0; i<TEST_ARRAY_SIZE; i++ )
partial_verify_vals[i] = key_array[test_index_array[i]];
for( i=0; i<MAX_KEY; i++ )
key_buff1[i] = 0;
}
#pragma omp barrier  
for (i=0; i<MAX_KEY; i++)
prv_buff1[i] = 0;
#pragma omp for nowait
for( i=0; i<NUM_KEYS; i++ ) {
key_buff2[i] = key_array[i];
prv_buff1[key_buff2[i]]++;  
}
for( i=0; i<MAX_KEY-1; i++ )   
prv_buff1[i+1] += prv_buff1[i];  
#pragma omp critical
{
for( i=0; i<MAX_KEY; i++ )
key_buff1[i] += prv_buff1[i];
}
#pragma omp barrier    
#pragma omp master
{
for( i=0; i<TEST_ARRAY_SIZE; i++ )
{                                             
k = partial_verify_vals[i];          
if( 0 <= k  &&  k <= NUM_KEYS-1 )
switch( CLASS )
{
case 'S':
if( i <= 2 )
{
if( key_buff1[k-1] != test_rank_array[i]+iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
else
{
if( key_buff1[k-1] != test_rank_array[i]-iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
break;
case 'W':
if( i < 2 )
{
if( key_buff1[k-1] != 
test_rank_array[i]+(iteration-2) )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
else
{
if( key_buff1[k-1] != test_rank_array[i]-iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
break;
case 'A':
if( i <= 2 )
{
if( key_buff1[k-1] != 
test_rank_array[i]+(iteration-1) )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
else
{
if( key_buff1[k-1] != 
test_rank_array[i]-(iteration-1) )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
break;
case 'B':
if( i == 1 || i == 2 || i == 4 )
{
if( key_buff1[k-1] != test_rank_array[i]+iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
else
{
if( key_buff1[k-1] != test_rank_array[i]-iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
break;
case 'C':
if( i <= 2 )
{
if( key_buff1[k-1] != test_rank_array[i]+iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
else
{
if( key_buff1[k-1] != test_rank_array[i]-iteration )
{
printf( "Failed partial verification: "
"iteration %d, test key %d\n", 
iteration, i );
}
else
passed_verification++;
}
break;
}        
}
if( iteration == MAX_ITERATIONS ) 
key_buff_ptr_global = key_buff1;
} 
}      
main( argc, argv )
int argc;
char **argv;
{
int             i, iteration, itemp;
int		    nthreads = 1;
double          timecounter, maxtime;
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
};
printf( "\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"
" - IS Benchmark\n\n" );
printf( " Size:  %d  (class %c)\n", TOTAL_KEYS, CLASS );
printf( " Iterations:   %d\n", MAX_ITERATIONS );
timer_clear( 0 );
create_seq( 314159265.00,                    
1220703125.00 );                 
#pragma omp parallel    
rank( 1 );  
passed_verification = 0;
if( CLASS != 'S' ) printf( "\n   iteration\n" );
timer_start( 0 );
#pragma omp parallel private(iteration)    
for( iteration=1; iteration<=MAX_ITERATIONS; iteration++ )
{
#pragma omp master	
if( CLASS != 'S' ) printf( "        %d\n", iteration );
rank( iteration );
#if defined(_OPENMP)	
#pragma omp master
nthreads = omp_get_num_threads();
#endif 	
}
timer_stop( 0 );
timecounter = timer_read( 0 );
full_verify();
if( passed_verification != 5*MAX_ITERATIONS + 1 )
passed_verification = 0;
c_print_results( "IS",
CLASS,
TOTAL_KEYS,
0,
0,
MAX_ITERATIONS,
nthreads,
timecounter,
((double) (MAX_ITERATIONS*TOTAL_KEYS))
/timecounter/1000000.,
"keys ranked", 
passed_verification,
NPBVERSION,
COMPILETIME,
CC,
CLINK,
C_LIB,
C_INC,
CFLAGS,
CLINKFLAGS,
"randlc");
}        
