#include <cassert>
#include <chrono>
#include <string>
#include <sstream>
#include "OptionParser.h"
#include "S3D.h"

using namespace std;

template <class real>
void RunTest(string testName, OptionParser &op);

template<class T> inline string toString(const T& t)
{
stringstream ss;
ss << t;
return ss.str();
}

void
addBenchmarkSpecOptions(OptionParser &op)
{
; 
}

void RunBenchmark(OptionParser &op)
{
auto t1 = std::chrono::high_resolution_clock::now();
RunTest<float>("S3D-SP", op);
auto t2 = std::chrono::high_resolution_clock::now();
double total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
printf("Total time %lf secs \n", total_time * 1e-9);

t1 = std::chrono::high_resolution_clock::now();
RunTest<float>("S3D-DP", op);
t2 = std::chrono::high_resolution_clock::now();
total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
printf("Total time %lf secs \n", total_time * 1e-9);
}

template <class real>
void RunTest(string testName, OptionParser &op)
{
int probSizes_SP[4] = { 8, 16, 32, 64 };
int probSizes_DP[4] = { 8, 16, 32, 64 };
int *probSizes = (sizeof(real) == sizeof(double)) ? probSizes_DP : probSizes_SP;
int sizeClass = op.getOptionInt("size") - 1;
assert(sizeClass >= 0 && sizeClass < 4);
sizeClass = probSizes[sizeClass];
int n = sizeClass * sizeClass * sizeClass;

real* host_t = (real*) malloc (n*sizeof(real));
real* host_p = (real*) malloc (n*sizeof(real));
real* host_y = (real*) malloc (Y_SIZE*n*sizeof(real));
real* host_molwt = (real*) malloc (WDOT_SIZE*sizeof(real));

real* RF = (real*) malloc (RF_SIZE*n*sizeof(real));
real* RB = (real*) malloc (RB_SIZE*n*sizeof(real));
real* RKLOW = (real*) malloc (RKLOW_SIZE*n*sizeof(real));
real* C = (real*) malloc (C_SIZE*n*sizeof(real));
real* A = (real*) malloc (A_SIZE*n*sizeof(real));
real* EG = (real*) malloc (EG_SIZE*n*sizeof(real));
real* WDOT = (real*) malloc (WDOT_SIZE*n*sizeof(real));


real rateconv = 1.0;
real tconv = 1.0;
real pconv = 1.0;

for (int i=0; i<n; i++)
{
host_p[i] = 1.0132e6;
host_t[i] = 1000.0;
}

for (int i=0; i<WDOT_SIZE; i++)
{
host_molwt[i] = 1;
}

for (int j=0; j<Y_SIZE; j++)
{
for (int i=0; i<n; i++)
{
host_y[(j*n)+i]= 0.0;
if (j==14)
host_y[(j*n)+i] = 0.064;
if (j==3)
host_y[(j*n)+i] = 0.218;
if (j==21)
host_y[(j*n)+i] = 0.718;
}
}

real *T = host_t;
real *P = host_p;
real *Y = host_y;
real *molwt = host_molwt;

int thrds = BLOCK_SIZE;
int thrds2 = BLOCK_SIZE2;

unsigned int passes = op.getOptionInt("passes");

#pragma omp target data map(to: T[0:n], P[0:n], Y[0:Y_SIZE*n], molwt[0:WDOT_SIZE]) \
map(alloc:RF[0:RF_SIZE*n], \
RB[0:RB_SIZE*n], \
RKLOW[0:RKLOW_SIZE*n], \
C[0:C_SIZE*n], \
A[0:A_SIZE*n], \
EG[0:EG_SIZE*n]) \
map(from: WDOT[0:WDOT_SIZE*n])
{
auto start = std::chrono::high_resolution_clock::now();

for (unsigned int i = 0; i < passes; i++)
{
#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "rdsmh.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "gr_base.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt2.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt3.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt4.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt5.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt6.h"
}
#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt7.h"
}
#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt8.h"
}
#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt9.h"
}
#pragma omp target teams distribute parallel for thread_limit(thrds2)
for (int i = 0; i < n; i++) {
#include "ratt10.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds)
for (int i = 0; i < n; i++) {
#include "ratx.h"
}

#pragma omp target teams distribute parallel for thread_limit(thrds)
for (int i = 0; i < n; i++) {
#include "ratxb.h"
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "ratx2.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "ratx4.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "qssa.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "qssab.h"
}
}
#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "qssa2.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot2.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot3.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot6.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot7.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot8.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot9.h"
}
}

#pragma omp target teams num_teams(n/thrds2) thread_limit(thrds2)
{
#pragma omp parallel
{
#include "rdwdot10.h"
}
}
}
auto end  = std::chrono::high_resolution_clock::now();
auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
printf("\nAverage time of executing s3d kernels: %lf (us)\n", (time * 1e-3) / passes);
}

for (int i=0; i<WDOT_SIZE; i++) {
printf("% 23.16E ", WDOT[i*n]);
if (i % 3 == 2)
printf("\n");
}
printf("\n");

free(host_t);
free(host_p);
free(host_y);
free(host_molwt);
free(RF);
free(RB);
free(RKLOW);
free(C);
free(A);
free(EG);
free(WDOT);
}
