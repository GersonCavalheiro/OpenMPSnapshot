


#include<iostream>
#include<ff/farm.hpp>

#if !defined(FF_TPC)
#define FF_TPC
#endif

#include <ff/tpcnode.hpp>
using namespace ff;

#define KERNEL1_ID	  1
#define MAX_SIZE 	512



struct Task: public baseTPCTask<Task> {
Task():in(nullptr),sizein(0),start(0),stop(0),result(0) {}

Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop):
in(in),sizein(sizein),start(start),stop(stop),result(0) {}

void setTask(const Task *t) { 

setKernelId(KERNEL1_ID);

setInPtr(&t->start, 1, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
setInPtr(&t->stop,  1,
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

setInPtr(t->in, t->sizein, 
first_time_flag?BitFlags::COPYTO:BitFlags::DONTCOPYTO, 
!first_time_flag?BitFlags::REUSE:BitFlags::DONTREUSE, 
BitFlags::DONTRELEASE);

setOutPtr(&t->result, 1, 
BitFlags::COPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

first_time_flag = false;
}

uint32_t *in;
uint32_t  sizein;
uint32_t  start, stop; 
uint32_t  result;

bool first_time_flag = true;
};


static inline
uint32_t gauss(uint32_t const to) {
return (to * to + to) / 2;
}

static inline
uint32_t ingauss(uint32_t const from, uint32_t to) {
return gauss(to) - gauss(from);
}


int main() {
const size_t size = 256;
uint32_t waits[size];
for (int j = 0; j < size; ++j)
waits[j] = j + 1;

struct Scheduler: ff_node_t<Task> {        
Scheduler(uint32_t *waits, size_t size):waits(waits),size(size) {}
Task *svc(Task *) {
for(int i=10;i<200;++i)
ff_send_out(new Task(waits, size, i, i+50));
return EOS;                
}
uint32_t *waits;
size_t    size;
} sched(waits, size);

struct Checker: ff_node_t<Task> {
Task *svc(Task *in) {
if (in->result != ingauss(in->start, in->stop+1))
std::cerr << "Wrong result: " << in->result << " (expected: " 
<< ingauss(in->start, in->stop+1) << ")\n"; 
else
std::cout << "RESULT OK " << in->result << "\n";
return GO_ON;
}
} checker;

ff_tpcallocator alloc;

ff_Farm<> farm([&]() {
const size_t nworkers = 4;
std::vector<std::unique_ptr<ff_node> > W;
for(size_t i=0;i<nworkers;++i)
W.push_back(make_unique<ff_tpcNode_t<Task> >(&alloc));
return W;
} (), sched, checker);

if (farm.run_and_wait_end()<0) {
error("running farm\n");
return -1;
}

return 0;
}

