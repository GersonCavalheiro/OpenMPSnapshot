


#include<iostream>

#if !defined(FF_TPC)
#define FF_TPC
#endif

#include <ff/tpcnode.hpp>
using namespace ff;

#define KERNEL1_ID	  1
#define MAX_SIZE 	512



struct Task: public baseTPCTask<Task> {
Task():in(nullptr),sizein(0),start(0),stop(0),result(0) {}

Task(uint32_t *in, uint32_t sizein, uint32_t start, uint32_t stop, uint32_t *result):
in(in),sizein(sizein),start(start),stop(stop),result(result) {}

void setTask(const Task *t) { 

setKernelId(KERNEL1_ID);
setInPtr(&t->start, 1);
setInPtr(&t->stop,  1);
setInPtr(t->in, t->sizein);
setOutPtr(t->result, 1);
}

uint32_t *in;
uint32_t  sizein;
uint32_t  start, stop; 
uint32_t *result;
};



static inline
uint32_t gauss(uint32_t const to) {
return (to * to + to) / 2;
}

static inline
uint32_t ingauss(uint32_t const from, uint32_t to) {
return gauss(to) - gauss(from);
}


int main(int argc, char * argv[]) {
const size_t size = 256;
uint32_t waits[size];
for (int j = 0; j < size; ++j)
waits[j] = j + 1;

uint32_t result = 0;

Task tpct(waits, size, 45, 69, &result);
ff_tpcNode_t<Task> tpcmap(tpct);

if (tpcmap.run_and_wait_end()<0) {
error("running tpcmap\n");
return -1;
}

if (result != ingauss(45, 70))
std::cerr << "Wrong return value: " << result << " (expected: " << ingauss(45, 70) << ")\n";
std::cout << "result = " << result << std::endl;

return 0;
}

