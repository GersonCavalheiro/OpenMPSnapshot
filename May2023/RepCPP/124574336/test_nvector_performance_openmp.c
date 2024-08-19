

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include <sundials/sundials_types.h>
#include <nvector/nvector_openmp.h>
#include <sundials/sundials_math.h>
#include "test_nvector_performance.h"


static int InitializeClearCache(int cachesize);
static int FinalizeClearCache();


static sunindextype N;  
static realtype* data;  



int main(int argc, char *argv[])
{
SUNContext   ctx = NULL;  
N_Vector     X   = NULL;  
sunindextype veclen;      

int print_timing;    
int ntests;          
int nvecs;           
int nsums;           
int cachesize;       
int nthreads;        
int flag;            

printf("Start Tests\n");
printf("Vector Name: OpenMP\n");


if (argc < 7){
printf("ERROR: SIX (6) arguments required: ");
printf("<vector length> <number of vectors> <number of sums> <number of tests> ");
printf("<cachesize (MB)> <print timing>\n");
return(-1);
}

veclen = (sunindextype) atol(argv[1]);
if (veclen <= 0) {
printf("ERROR: length of vector must be a positive integer \n");
return(-1);
}

nvecs = (int) atol(argv[2]);
if (nvecs <= 0) {
printf("ERROR: number of vectors must be a positive integer \n");
return(-1);
}

nsums = (int) atol(argv[3]);
if (nsums <= 0) {
printf("ERROR: number of sums must be a positive integer \n");
return(-1);
}

ntests = (int) atol(argv[4]);
if (ntests <= 0) {
printf("ERROR: number of tests must be a positive integer \n");
return(-1);
}

cachesize = (int) atol(argv[5]);
if (cachesize < 0) {
printf("ERROR: cache size (MB) must be a non-negative integer \n");
return(-1);
}
InitializeClearCache(cachesize);

print_timing = atoi(argv[6]);
SetTiming(print_timing, 0);

#pragma omp parallel
{
#pragma omp single
nthreads = omp_get_num_threads();
}

printf("\nRunning with: \n");
printf("  vector length         %ld \n", (long int) veclen);
printf("  max number of vectors %d  \n", nvecs);
printf("  max number of sums    %d  \n", nsums);
printf("  number of tests       %d  \n", ntests);
printf("  timing on/off         %d  \n", print_timing);
printf("  number of threads     %d  \n", nthreads);

flag = SUNContext_Create(NULL, &ctx);
if (flag) return flag;


X = N_VNew_OpenMP(veclen, nthreads, ctx);


if (print_timing) printf("\n\n standard operations:\n");
if (print_timing) PrintTableHeader(1);
flag = Test_N_VLinearSum(X, veclen, ntests);
flag = Test_N_VConst(X, veclen, ntests);
flag = Test_N_VProd(X, veclen, ntests);
flag = Test_N_VDiv(X, veclen, ntests);
flag = Test_N_VScale(X, veclen, ntests);
flag = Test_N_VAbs(X, veclen, ntests);
flag = Test_N_VInv(X, veclen, ntests);
flag = Test_N_VAddConst(X, veclen, ntests);
flag = Test_N_VDotProd(X, veclen, ntests);
flag = Test_N_VMaxNorm(X, veclen, ntests);
flag = Test_N_VWrmsNorm(X, veclen, ntests);
flag = Test_N_VWrmsNormMask(X, veclen, ntests);
flag = Test_N_VMin(X, veclen, ntests);
flag = Test_N_VWL2Norm(X, veclen, ntests);
flag = Test_N_VL1Norm(X, veclen, ntests);
flag = Test_N_VCompare(X, veclen, ntests);
flag = Test_N_VInvTest(X, veclen, ntests);
flag = Test_N_VConstrMask(X, veclen, ntests);
flag = Test_N_VMinQuotient(X, veclen, ntests);

if (print_timing) printf("\n\n fused operations 1: nvecs= %d\n", nvecs);
if (print_timing) PrintTableHeader(2);
flag = Test_N_VLinearCombination(X, veclen, nvecs, ntests);
flag = Test_N_VScaleAddMulti(X, veclen, nvecs, ntests);
flag = Test_N_VDotProdMulti(X, veclen, nvecs, ntests);
flag = Test_N_VLinearSumVectorArray(X, veclen, nvecs, ntests);
flag = Test_N_VScaleVectorArray(X, veclen, nvecs, ntests);
flag = Test_N_VConstVectorArray(X, veclen, nvecs, ntests);
flag = Test_N_VWrmsNormVectorArray(X, veclen, nvecs, ntests);
flag = Test_N_VWrmsNormMaskVectorArray(X, veclen, nvecs, ntests);

if (print_timing) printf("\n\n fused operations 2: nvecs= %d nsums= %d\n",
nvecs, nsums);
if (print_timing) PrintTableHeader(2);
flag = Test_N_VScaleAddMultiVectorArray(X, veclen, nvecs, nsums, ntests);
flag = Test_N_VLinearCombinationVectorArray(X, veclen, nvecs, nsums, ntests);


N_VDestroy(X);

FinalizeClearCache();

flag = SUNContext_Free(&ctx);
if (flag) return flag;

printf("\nFinished Tests\n");

return(flag);
}





void N_VRand(N_Vector Xvec, sunindextype Xlen, realtype lower, realtype upper)
{
realtype *Xdata;

Xdata = N_VGetArrayPointer(Xvec);
rand_realtype(Xdata, Xlen, lower, upper);
}


void N_VRandZeroOne(N_Vector Xvec, sunindextype Xlen)
{
realtype *Xdata;

Xdata = N_VGetArrayPointer(Xvec);
rand_realtype_zero_one(Xdata, Xlen);
}


void N_VRandConstraints(N_Vector Xvec, sunindextype Xlen)
{
realtype *Xdata;

Xdata = N_VGetArrayPointer(Xvec);
rand_realtype_constraints(Xdata, Xlen);
}




void collect_times(N_Vector X, double *times, int ntimes)
{

return;
}

void sync_device(N_Vector x)
{

return;
}




static int InitializeClearCache(int cachesize)
{
size_t nbytes;  


nbytes = (size_t) (2 * cachesize * 1024 * 1024);
N = (sunindextype) ((nbytes + sizeof(realtype) - 1)/sizeof(realtype));


data = (realtype*) malloc(N*sizeof(realtype));
rand_realtype(data, N, RCONST(-1.0), RCONST(1.0));

return(0);
}

static int FinalizeClearCache()
{
free(data);
return(0);
}

void ClearCache()
{
realtype     sum;
sunindextype i;

sum = RCONST(0.0);
for (i=0; i<N; i++)
sum += data[i];

return;
}
