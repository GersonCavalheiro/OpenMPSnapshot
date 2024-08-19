



#include <functional>
#include <algorithm>
#include <vector>
#include <ff/node.hpp>
#include <ff/barrier.hpp>

using namespace ff;

#if defined(TEST_OMP_BARRIER)
#include <omp.h>
#endif

#if defined(USE_PTHREAD_BARRIER)
#define SPIN 0
#else
#define SPIN 1
#endif
static barrierSelector<SPIN> bar;

int* data=NULL;

#define BARRIER   bar.doBarrier(get_my_id())

class Thread: public ff_node {
public:
Thread(int nbarriers, bool check=true):nbarriers(nbarriers),check(check) {}

#if 0
int svc_init() {
printf("thread %d running on cpu %d\n", get_my_id(), ff_getMyCore());
return 0;
}
#endif

void* svc(void*) {
for(int i=0;i<nbarriers;++i) {
++data[get_my_id()];
BARRIER;
if (check) {
assert(data[get_my_id()]==data[0]);
BARRIER;
}
}
return EOS;
}
void set_id(ssize_t id) {
ff_node::set_id(id);
}
int run(bool=false)  { return ff_node::run();}
int wait()           { return ff_node::wait();}

double wffTime() {
return ff_node::wffTime();
}

private:
int nbarriers;
bool check;
};

int main(int argc, char* argv[]) {
int nthreads = 17;
int nbarriers= 15;
if (argc>1) {
if (argc < 3) {
printf("use: %s num-threads num-barriers\n", argv[0]);
return -1;
}
nthreads=atoi(argv[1]);
assert(nthreads>0);
nbarriers=atoi(argv[2]);
}
bar.barrierSetup(nthreads);
data = (int*)malloc(sizeof(int)*nthreads);
assert(data);

Thread** N = new Thread*[nthreads];

for(int i=0;i<nthreads;++i) {
data[i]=0;
N[i]= new Thread(nbarriers);
assert(N[i]);
N[i]->set_id(i);
N[i]->run();
}
for(int i=0;i<nthreads;++i) 
N[i]->wait();

std::vector<double > threadTime(nthreads,0.0);
for(int i=0;i<nthreads;++i)
threadTime[i]=N[i]->wffTime();

std::vector<double >::iterator it=
std::max_element(threadTime.begin(),threadTime.end(),std::less<double>() );

printf("Time %.02f (ms)\n", *it);

#if defined(TEST_OMP_BARRIER)
for(int i=0;i<nthreads;++i) {
data[i]=0;
}
omp_set_num_threads(nthreads);

ffTime(START_TIME);
#pragma omp parallel
for(int i=0;i<nbarriers;++i) {
++data[omp_get_thread_num()];
#pragma omp barrier
if (true) {
assert(data[omp_get_thread_num()]==data[0]);
#pragma omp barrier
}
}
ffTime(STOP_TIME);
printf("openmp Time %.02f (ms)\n", ffTime(GET_TIME));
#endif
return 0;
}
