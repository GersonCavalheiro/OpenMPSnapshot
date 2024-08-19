




#include <cstdlib>
#if defined(USE_OPENMP)
#include <omp.h>
#endif

#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

#if defined(USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif


using namespace ff;

int main(int argc, char *argv[]) {
int  nworkers = 3;
long numtasks = 10000000*nworkers;
int   chunk = 100;
if (argc>1) {
if (argc<3) {
printf("use: %s numtasks nworkers [chunk=(numtasks/nworkers)]\n", argv[0]);
return -1;
}
numtasks = atol(argv[1]);
nworkers = atoi(argv[2]);
if (argc == 4) 
chunk = atoi(argv[3]);
}

long *V;
if (posix_memalign((void**)&V, 64, numtasks*sizeof(long)) != 0) abort();

#if defined(USE_OPENMP)
ffTime(START_TIME);
#pragma omp parallel for schedule(runtime) num_threads(nworkers)
for(long j=0;j<numtasks;++j) {
V[j]=j;
}
ffTime(STOP_TIME);
printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#elif defined(USE_TBB)

tbb::task_scheduler_init init(nworkers);
tbb::affinity_partitioner ap;

ffTime(START_TIME);
tbb::parallel_for(tbb::blocked_range<long>(0, numtasks, chunk),
[&] (const tbb::blocked_range<long>& r) {
for (long j=r.begin();j!=r.end();++j) {
V[j] = j;
}
}, ap);
ffTime(STOP_TIME);
printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#else

#if 0
FF_PARFOR_INIT(pf, nworkers);
ffTime(START_TIME);
FF_PARFOR_START(pf, j,0,numtasks,1, chunk, nworkers) {
V[j]=j;
} FF_PARFOR_STOP(pf);
ffTime(STOP_TIME);
printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
FF_PARFOR_DONE(pf);
#else
ParallelFor pf(nworkers);
ffTime(START_TIME);
pf.parallel_for(0,numtasks,1,chunk, [&V](const long j) {
V[j]=j;
});
ffTime(STOP_TIME);
printf("%d Time  = %g (ms)\n", nworkers, ffTime(GET_TIME));
#endif
#endif
return 0;
}
