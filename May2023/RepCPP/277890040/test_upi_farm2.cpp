


#include<iostream>
#include<ff/farm.hpp>

#if !defined(FF_TPC)
#define FF_TPC
#endif

#include <ff/tpcnode.hpp>
using namespace ff;

#define KERNEL1_ID	  2
#define KERNEL2_ID    1
#define MAX_SIZE 	512


struct TaskCopy: public baseTPCTask<TaskCopy> {
TaskCopy():in(nullptr),out(nullptr), sizein(0), sizeout(0) {}

TaskCopy(uint32_t *in, uint32_t sizein, uint32_t *out, uint32_t sizeout):
in(in),out(out),sizein(sizein),sizeout(sizeout) {}

void setTask(const TaskCopy *t) { 

setKernelId(KERNEL1_ID);

setInPtr(&t->sizein, 1, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::RELEASE);
setInPtr(t->in, t->sizein, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
setInPtr(&t->sizeout, 1, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::RELEASE);
setOutPtr(t->out, t->sizeout, 
BitFlags::DONTCOPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
}

uint32_t *in,*out;
uint32_t  sizein, sizeout;
};


struct Task: public baseTPCTask<Task> {
Task():in(nullptr),sizein(0),start(0),stop(0),result(0) {}

Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop):
in(in),sizein(sizein),start(start),stop(stop),result(0) {}

void setTask(const Task *t) { 

setKernelId(KERNEL2_ID);

setInPtr(&t->start, 1, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);
setInPtr(&t->stop,  1, 
BitFlags::COPYTO, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

setInPtr(t->in, t->sizein, 
BitFlags::DONTCOPYTO, BitFlags::REUSE, BitFlags::DONTRELEASE);

setOutPtr(&t->result, 1, 
BitFlags::COPYBACK, BitFlags::DONTREUSE, BitFlags::DONTRELEASE);

}

uint32_t *in;
uint32_t  sizein;
uint32_t  start, stop; 
uint32_t  result;
};




static inline
uint32_t gauss(uint32_t const to) {
return (to * to + to) / 2;
}

static inline
uint32_t ingauss(uint32_t const from, uint32_t to) {
return gauss(to) - gauss(from);
}

static 
void check(uint32_t to, uint32_t from, uint32_t result) {
if (result != ingauss(to, from+1))
std::cerr << "Wrong result: " << result << " (expected: " 
<< ingauss(to, from+1) << ")\n"; 
else
std::cout << "RESULT OK " << result << "\n";    
}



int main() {
const size_t size = 256;
uint32_t waits[size];
uint32_t waits2[size] {0};
for (int j = 0; j < size; ++j)
waits[j] = j + 1;

ff_tpcallocator alloc;


TaskCopy k1(waits, size, waits2, size);
ff_tpcNode_t<TaskCopy> copy(k1, &alloc);



struct Scheduler: ff_node_t<Task> {        
Scheduler(uint32_t *waits, size_t size):waits(waits),size(size) {}
Task *svc(Task *) {
for(int i=10;i<120;++i)
ff_send_out(new Task(waits, size, i, i+50));
return EOS;                
}
uint32_t *waits;
size_t    size;
} sched(waits2, size);

struct Checker: ff_node_t<Task> {
Task *svc(Task *in) {
check(in->start, in->stop, in->result);
return GO_ON;
}
} checker;


ff_Farm<> farm([&]() {
const size_t nworkers = 4;
std::vector<std::unique_ptr<ff_node> > W;
for(size_t i=0;i<nworkers;++i)
W.push_back(make_unique<ff_tpcNode_t<Task> >(&alloc));
return W;
} (), sched, checker);



if (copy.run_and_wait_end()<0) {
error("running first kernel\n");
return -1;        
}
if (farm.run_and_wait_end()<0) {
error("running farm\n");
return -1;
}

return 0;
}

