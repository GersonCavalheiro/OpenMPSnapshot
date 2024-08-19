#include <omp.h> 
#include "vSched.h"
double constraint;
#define THREE_POINT_OH 3.0 
#define FORALL_BEGIN(strat, s,e, start, end, tid, numThds ) loop_start_ ## strat (s,e ,&start, &
end, tid, numThds); do {
#define FORALL_END(strat, start, end, tid) } while( loop_next_ ## strat (&start, &end, tid));
int main(int argc, char* argv[]) {
int timestep = 0;
int numThrds;
int start, end = 0;
double fd, fs = 0.0;
static LoopTimeRecord *record = NULL;
#pragma omp single
numThreads = omp_get_num_threads(); 
vSched_init(numThreads); 
if (argc < 1)
printf("Usage: app [numThreads]\n");
else
{
numThrds = atoi(argv[1]); 
omp_set_num_threads(numThreads);
}
while(timestep < 1000) {
fs = 1.0 - fd;
#pragma omp parallel
{
int tid = omp_get_thread_num();
int numThrds = omp_get_num_threads();
FORALL_BEGIN(sds,tid,numThrds, 0, n, start, end, fs)
for(int i=start;i<end;i++)
b[i] = (a[i] + a[i+1] + a[i-1])/THREE_POINT_OH;
FORALL_END(sds,tid,start,end)
}
temp = b; 
b = a; 
a = temp;
}
vSched_finalize(numThrds);
}
