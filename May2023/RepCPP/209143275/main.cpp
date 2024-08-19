#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <omp.h>
#include "Stack.h"
#include "dummy.h"

using namespace std;
#define DEFINE_ARR(an, sz)		double *an = NULL; an = new double [sz]; for(int i=0;i<sz;i++) an[i]=0.; assert(an)

Stackc a1_STACKc;
Stacki a1_STACKi;
Stackf a1_STACKf;
#pragma omp threadprivate (a1_STACKc, a1_STACKi, a1_STACKf)

int main(int argc, char* argv[])
{
int p=omp_get_max_threads();
int n;
DEFINE_ARR(x,100);
cerr << "test: ";
x[0]=1.;
x[0]+=x[0];
x[0]+=x[0];
cerr << x[0] << endl;
#if 0
#pragma omp parallel num_threads(2)
{
int tid;
double y;
tid=omp_get_thread_num();
a1_STACKf.init();
if(tid==0) { x[0]=1.; }
#pragma omp barrier
a1_STACKf.push(x[0]);
if(tid==0) usleep(1e6);
y=x[0];
#pragma omp atomic
x[0]+=x[0];
#pragma omp barrier
assert(a1_STACKf.top()==y);
assert(x[0]==4.);
dummy("", x);
}
#endif
#if 1
cerr << "Another try:\n";
#pragma omp parallel num_threads(2)
{
int tid;
double y;
tid=omp_get_thread_num();
a1_STACKf.init();
if(tid==0) { x[0]=1.; }
#pragma omp barrier
a1_STACKf.push(x[0]);
#pragma omp atomic
x[0]+=x[0];
#pragma omp barrier
#pragma omp master
{
cerr << x[0] << endl;
assert(x[0]==4.);
}
#pragma omp atomic
x[0]+=-a1_STACKf.top();
#pragma omp barrier
#pragma omp master
{
cerr << x[0] << endl;
assert(x[0]==1.);
}
dummy("", x);
}
#endif
#if 0
cerr << "Another try:\n";
#pragma omp parallel num_threads(2)
{
int tid;
double y;
tid=omp_get_thread_num();
a1_STACKf.init();
if(tid==0) { x[0]=1.; }
#pragma omp barrier
#pragma omp critical
{
a1_STACKf.push(x[0]);
if(tid==0) usleep(1e6);
y=x[0];
#pragma omp atomic
x[0]+=x[0];
}
#pragma omp barrier
assert(a1_STACKf.top()==y);
assert(x[0]==4.);
dummy("", x);
}
#endif
return 0;
}


