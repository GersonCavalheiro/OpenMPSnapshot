#include "npbparams.h"
#include <stdlib.h>
#include <stdio.h>
typedef int INT_TYPE;
INT_TYPE * key_buff_ptr_global;
int passed_verification;
INT_TYPE key_array[(1<<23)], key_buff1[(1<<23)], key_buff2[(1<<23)], partial_verify_vals[5];
INT_TYPE test_index_array[5], test_rank_array[5], S_test_index_array[5] = {48427, 17148, 23627, 62548, 4431}, S_test_rank_array[5] = {0, 18, 346, 64917, 65463}, W_test_index_array[5] = {357773, 934767, 875723, 898999, 404505}, W_test_rank_array[5] = {1249, 11698, 1039987, 1043896, 1048018}, A_test_index_array[5] = {2112377, 662041, 5336171, 3642833, 4250760}, A_test_rank_array[5] = {104, 17523, 123928, 8288932, 8388264}, B_test_index_array[5] = {41869, 812306, 5102857, 18232239, 26860214}, B_test_rank_array[5] = {33422937, 10244, 59149, 33135281, 99}, C_test_index_array[5] = {44172927, 72999161, 74326391, 129606274, 21736814}, C_test_rank_array[5] = {61147, 882988, 266290, 133997595, 133525895};
double randlc(double * X, double * A);
void full_verify(void );
double randlc(double * X, double * A)
{
static int KS = 0;
static double R23, R46, T23, T46;
double T1, T2, T3, T4;
double A1;
double A2;
double X1;
double X2;
double Z;
int i, j;
double _ret_val_0;
if (KS==0)
{
R23=1.0;
R46=1.0;
T23=1.0;
T46=1.0;
#pragma loop name randlc#0 
#pragma cetus reduction(*: R23, T23) 
#pragma cetus parallel 
#pragma omp parallel for reduction(*: R23, T23)
for (i=1; i<=23; i ++ )
{
R23=(0.5*R23);
T23=(2.0*T23);
}
#pragma loop name randlc#1 
#pragma cetus reduction(*: R46, T46) 
#pragma cetus parallel 
#pragma omp parallel for reduction(*: R46, T46)
for (i=1; i<=46; i ++ )
{
R46=(0.5*R46);
T46=(2.0*T46);
}
KS=1;
}
T1=(R23*( * A));
j=T1;
A1=j;
A2=(( * A)-(T23*A1));
T1=(R23*( * X));
j=T1;
X1=j;
X2=(( * X)-(T23*X1));
T1=((A1*X2)+(A2*X1));
j=(R23*T1);
T2=j;
Z=(T1-(T23*T2));
T3=((T23*Z)+(A2*X2));
j=(R46*T3);
T4=j;
( * X)=(T3-(T46*T4));
_ret_val_0=(R46*( * X));
return _ret_val_0;
}
void create_seq(double seed, double a)
{
double x;
int i, j, k;
k=((1<<19)/4);
#pragma loop name create_seq#0 
for (i=0; i<(1<<23); i ++ )
{
x=randlc( & seed,  & a);
x+=randlc( & seed,  & a);
x+=randlc( & seed,  & a);
x+=randlc( & seed,  & a);
key_array[i]=(k*x);
}
return ;
}
void full_verify()
{
INT_TYPE i, j;
INT_TYPE k;
INT_TYPE m, unique_keys;
#pragma loop name full_verify#0 
for (i=0; i<(1<<23); i ++ )
{
key_array[ -- key_buff_ptr_global[key_buff2[i]]]=key_buff2[i];
}
j=0;
#pragma loop name full_verify#1 
#pragma cetus reduction(+: j) 
#pragma cetus parallel 
#pragma omp parallel for reduction(+: j)
for (i=1; i<(1<<23); i ++ )
{
if (key_array[i-1]>key_array[i])
{
j ++ ;
}
}
if (j!=0)
{
printf("Full_verify: number of keys out of sort: %d\n", j);
}
else
{
passed_verification ++ ;
}
return ;
}
void rank(int iteration)
{
INT_TYPE i, j, k;
INT_TYPE l, m;
INT_TYPE shift = 19-10;
INT_TYPE key;
INT_TYPE min_key_val, max_key_val;
INT_TYPE prv_buff1[(1<<19)];
key_array[iteration]=iteration;
key_array[iteration+10]=((1<<19)-iteration);
#pragma loop name rank#0 
for (i=0; i<5; i ++ )
{
partial_verify_vals[i]=key_array[test_index_array[i]];
}
#pragma loop name rank#1 
#pragma cetus parallel 
#pragma omp parallel for
for (i=0; i<(1<<19); i ++ )
{
key_buff1[i]=0;
}
#pragma loop name rank#2 
#pragma cetus parallel 
#pragma omp parallel for
for (i=0; i<(1<<19); i ++ )
{
prv_buff1[i]=0;
}
#pragma loop name rank#3 
#pragma cetus reduction(+: prv_buff1[key_buff2[i]]) 
for (i=0; i<(1<<23); i ++ )
{
key_buff2[i]=key_array[i];
prv_buff1[key_buff2[i]] ++ ;
}
#pragma loop name rank#4 
for (i=0; i<((1<<19)-1); i ++ )
{
prv_buff1[i+1]+=prv_buff1[i];
}
#pragma loop name rank#5 
for (i=0; i<(1<<19); i ++ )
{
key_buff1[i]+=prv_buff1[i];
}
#pragma loop name rank#6 
#pragma cetus reduction(+: passed_verification) 
for (i=0; i<5; i ++ )
{
k=partial_verify_vals[i];
if ((0<=k)&&(k<=((1<<23)-1)))
{
switch ('A')
{
case 'S':
if (i<=2)
{
if (key_buff1[k-1]!=(test_rank_array[i]+iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
else
{
if (key_buff1[k-1]!=(test_rank_array[i]-iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
break;
case 'W':
if (i<2)
{
if (key_buff1[k-1]!=(test_rank_array[i]+(iteration-2)))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
else
{
if (key_buff1[k-1]!=(test_rank_array[i]-iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
break;
case 'A':
if (i<=2)
{
if (key_buff1[k-1]!=(test_rank_array[i]+(iteration-1)))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
else
{
if (key_buff1[k-1]!=(test_rank_array[i]-(iteration-1)))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
break;
case 'B':
if (((i==1)||(i==2))||(i==4))
{
if (key_buff1[k-1]!=(test_rank_array[i]+iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
else
{
if (key_buff1[k-1]!=(test_rank_array[i]-iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
break;
case 'C':
if (i<=2)
{
if (key_buff1[k-1]!=(test_rank_array[i]+iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
else
{
if (key_buff1[k-1]!=(test_rank_array[i]-iteration))
{
printf("Failed partial verification: ""iteration %d, test key %d\n", iteration, i);
}
else
{
passed_verification ++ ;
}
}
break;
}
}
}
if (iteration==10)
{
key_buff_ptr_global=key_buff1;
}
return ;
}
main(int argc, char * * argv)
{
int i, iteration, itemp;
int nthreads = 1;
double timecounter, maxtime;
int _ret_val_0;
#pragma loop name main#0 
for (i=0; i<5; i ++ )
{
switch ('A')
{
case 'S':
test_index_array[i]=S_test_index_array[i];
test_rank_array[i]=S_test_rank_array[i];
break;
case 'A':
test_index_array[i]=A_test_index_array[i];
test_rank_array[i]=A_test_rank_array[i];
break;
case 'W':
test_index_array[i]=W_test_index_array[i];
test_rank_array[i]=W_test_rank_array[i];
break;
case 'B':
test_index_array[i]=B_test_index_array[i];
test_rank_array[i]=B_test_rank_array[i];
break;
case 'C':
test_index_array[i]=C_test_index_array[i];
test_rank_array[i]=C_test_rank_array[i];
break;
}
}
;
printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - IS Benchmark\n\n");
printf(" Size:  %d  (class %c)\n", 1<<23, 'A');
printf(" Iterations:   %d\n", 10);
timer_clear(0);
create_seq(3.14159265E8, 1.220703125E9);
rank(1);
passed_verification=0;
printf("\n   iteration\n");
timer_start(0);
#pragma loop name main#1 
for (iteration=1; iteration<=10; iteration ++ )
{
printf("        %d\n", iteration);
rank(iteration);
}
timer_stop(0);
timecounter=timer_read(0);
full_verify();
if (passed_verification!=((5*10)+1))
{
passed_verification=0;
}
c_print_results("IS", 'A', 1<<23, 0, 0, 10, nthreads, timecounter, (((double)(10*(1<<23)))/timecounter)/1000000.0, "keys ranked", passed_verification, "3.0 structured", "28 Nov 2019", "(none)", "(none)", "-lm", "(none)", "(none)", "(none)", "randlc");
return _ret_val_0;
}
