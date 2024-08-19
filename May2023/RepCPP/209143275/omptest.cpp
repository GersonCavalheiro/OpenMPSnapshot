#include <iostream>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <cassert>
#include "dummy.h"
using namespace std;

ulong fp_ops=0;

int main(int argc, char* argv[])
{
const double x[8] = {1,2,3,4,5,6,7,8};
double A[8*8] ;
for(int i=0;i<8;i++) for(int j=0;j<8;j++) A[i*8+j]=i*8+j+1;
double thread_result[4];
int n=8;
assert((n/2)%omp_get_max_threads()==0);
#pragma omp parallel
{
int tests=0;
int tid=0;
int p=0;
int i=0;
int j=0;
int lb=0;
int ub=0;
int c=0;
double y=0.;
tid=omp_get_thread_num();
p=omp_get_num_threads();
c=n/p;
lb=tid*c;
ub=(tid+1)*c-1;
i=lb;
y=1.;
while(i<=ub) {
j=0;
while(j<=n) {
#pragma omp atomic
fp_ops+=4;
y=y*sin(A[i*n+j]*x[i])*cos(A[i*n+j]*x[i]);
j=j+1;
}
i=i+1;
}
thread_result[tid]=y;
#pragma omp barrier
#pragma omp master
{
i=1;
while(i<p) {
#pragma omp atomic
fp_ops+=1;
thread_result[0]=thread_result[0]*thread_result[i];
i=i+1;
}
}
dummy("", (void*)thread_result);
}
cerr << thread_result[0] << endl;
cerr << "fp_ops: " << fp_ops << endl;
return 0;
}

