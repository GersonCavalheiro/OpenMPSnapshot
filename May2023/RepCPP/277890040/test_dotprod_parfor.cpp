




#if defined(TEST_PARFOR_PIPE_REDUCE)
#if !defined(HAS_CXX11_VARIADIC_TEMPLATES)
#define HAS_CXX11_VARIADIC_TEMPLATES 1
#endif
#endif
#include <ff/ff.hpp>
#include <ff/parallel_for.hpp>

#if defined(USE_TBB)
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>
#endif

using namespace ff;

int main(int argc, char * argv[]) {    
const double INITIAL_VALUE = 5.0;
int arraySize= 10000000;
int nworkers = 3;
int NTIMES   = 5;
int CHUNKSIZE= (std::min)(10000, arraySize/nworkers);
if (argc>1) {
if (argc<3) {
printf("use: %s arraysize nworkers [ntimes] [CHUNKSIZE]\n", argv[0]);
return -1;
}
arraySize= atoi(argv[1]);
nworkers = atoi(argv[2]);

if (argc>=4) NTIMES = atoi(argv[3]);
if (argc==5) CHUNKSIZE = atoi(argv[4]);
}

if (nworkers<=0) {
printf("Wrong parameters values\n");
return -1;
}


double *A = new double[arraySize];
double *B = new double[arraySize];

double sum = INITIAL_VALUE;
#if defined(USE_OPENMP)
#pragma omp parallel for schedule(runtime) num_threads(nworkers)
for(long j=0;j<arraySize;++j) {
A[j]=j*3.14; B[j]=2.1*j;
}

ff::ffTime(ff::START_TIME);
for(int z=0;z<NTIMES;++z) {

#pragma omp parallel for default(shared)                                \
schedule(runtime)                                                   \
reduction(+:sum)                                                    \
num_threads(nworkers)
for(long i=0;i<arraySize;++i)
sum += A[i]*B[i];

} 

ffTime(STOP_TIME);
printf("omp %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);

#elif defined(USE_TBB)

tbb::task_scheduler_init init(nworkers);
tbb::affinity_partitioner ap;

tbb::parallel_for(tbb::blocked_range<long>(0, arraySize, CHUNKSIZE),
[&] (const tbb::blocked_range<long> &r) {
for(long j=r.begin(); j!=r.end(); ++j) {
A[j]=j*3.14; B[j]=2.1*j;
}
},ap);

ff::ffTime(ff::START_TIME);
for(int z=0;z<NTIMES;++z) {
sum += tbb::parallel_reduce(tbb::blocked_range<long>(0, arraySize, CHUNKSIZE), double(0),
[=] (const tbb::blocked_range<long> &r, double in) {
return std::inner_product(A+r.begin(),
A+r.end(),
B+r.begin(),
in,
std::plus<double>(),
std::multiplies<double>());                                                              
}, std::plus<double>(), ap);

}
ffTime(STOP_TIME);
printf("tbb %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);

#else 

#if 1
#if defined(TEST_PARFOR_PIPE_REDUCE)
{        
parallel_for(0,arraySize,1,CHUNKSIZE, [&](const long j) { A[j]=j*3.14; B[j]=2.1*j;});
ParallelForPipeReduce<double*> pfr(nworkers, (nworkers < ff_numCores())); 

auto Map = [&](const long start, const long stop, const int thid, ff_buffernode &node) {
if (start == stop) return;
double localsum = 0.0;
for(long i=start;i<stop;++i)
localsum += A[i]*B[i];
node.put(new double(localsum));
};
auto Reduce = [&](const double* v) {
sum +=*v;
};

ff::ffTime(ff::START_TIME);    
for(int z=0;z<NTIMES;++z) {
pfr.parallel_reduce_idx(0, arraySize,1,CHUNKSIZE, Map, Reduce);
}
ffTime(STOP_TIME);
printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
}
#else 
{
ParallelForReduce<double> pfr(nworkers, (nworkers < ff_numCores())); 

pfr.parallel_for(0,arraySize,1,CHUNKSIZE, [&](const long j) { A[j]=j*3.14; B[j]=2.1*j;});
auto Fsum = [](double& v, const double& elem) { v += elem; };

ff::ffTime(ff::START_TIME);    
for(int z=0;z<NTIMES;++z) {
pfr.parallel_reduce(sum, 0.0, 
0, arraySize,1,CHUNKSIZE,
[&](const long i, double& sum) {sum += A[i]*B[i];}, 
Fsum); 
}
ffTime(STOP_TIME);
printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
}
#endif

#else  

FF_PARFORREDUCE_INIT(dp, double, nworkers);

FF_PARFOR_BEGIN(init, j,0,arraySize,1, CHUNKSIZE, nworkers) {
A[j]=j*3.14; B[j]=2.1*j;
} FF_PARFOR_END(init);


ff::ffTime(ff::START_TIME);    
for(int z=0;z<NTIMES;++z) {
FF_PARFORREDUCE_START(dp, sum, 0.0, i,0,arraySize,1, CHUNKSIZE, nworkers) { 
sum += A[i]*B[i];
} FF_PARFORREDUCE_STOP(dp, sum, +);    
}
ffTime(STOP_TIME);
printf("ff %d Time = %g ntimes=%d\n", nworkers, ffTime(GET_TIME), NTIMES);
FF_PARFORREDUCE_DONE(dp);
#endif 

#endif

printf("Sum =%g\n", sum);
return 0;
}
