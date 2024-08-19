



#include <cstdint>
#include <omp.h>
#include <immintrin.h>

#include "cpuid.h"
#include "barrier.hpp"

#include <new>

using namespace std;

namespace synk {

static Barrier *instance = NULL;

void Barrier::initializeInstance(int numCores, int numThreadsPerCore) {
if (!instance) {
if (omp_in_parallel()) {
#pragma omp barrier
#pragma omp master
instance = new Barrier(numCores, numThreadsPerCore);
#pragma omp barrier
instance->init(omp_get_thread_num());
} else {
instance = new Barrier(numCores, numThreadsPerCore);
#pragma omp parallel
{
instance->init(omp_get_thread_num());
}
}
}
}

Barrier *Barrier::getInstance() {
if (!instance) {
int nthreads = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();

#ifdef __MIC__
int threadsPerCore = nthreads%4 == 0 ? 4 : (nthreads%2 == 0 ? 2 : 1);
#else
int threadsPerCore = 1;
#endif

if (omp_in_parallel()) {
#pragma omp barrier
#pragma omp master
instance = new Barrier(nthreads / threadsPerCore, threadsPerCore);
#pragma omp barrier
instance->init(omp_get_thread_num());
} else {
instance = new Barrier(nthreads / threadsPerCore, threadsPerCore);
#pragma omp parallel
{
instance->init(omp_get_thread_num());
}
}
}
return instance;
}

void Barrier::deleteInstance() {
delete instance;
instance = NULL;
}


Barrier::Barrier(int numCores_, int numThreadsPerCore_)
: Synk(numCores_, numThreadsPerCore_) {
cores = (CoreBarrier **)
_mm_malloc(numCores * sizeof(CoreBarrier *), 64);
if (cores == NULL) throw bad_alloc();
threadCores = (CoreBarrier **)
_mm_malloc(numThreads * sizeof(CoreBarrier *), 64);
if (threadCores == NULL) throw bad_alloc();
coreTids = (int8_t *)
_mm_malloc(numThreads * sizeof(int8_t), 64);
if (coreTids == NULL) throw bad_alloc();
}



void Barrier::init(int tid) {
CoreBarrier *core;


int cid = tid / numThreadsPerCore;
int coreTid = tid % numThreadsPerCore;


if (coreTid == 0) {
core = (CoreBarrier *) _mm_malloc(sizeof(CoreBarrier), 64);
core->coreId = cid;
core->coreSense = 1;
core->threadSenses = (uint8_t *)
_mm_malloc(numThreadsPerCore * sizeof(uint8_t), 64);
for (int i = 0; i < numThreadsPerCore; i++)
core->threadSenses[i] = 1;
for (int i = 0; i < 2; i++) {
core->myFlags[i] = (uint8_t *)
_mm_malloc(lgNumCores * CacheLineSize, 64);
for (int j = 0; j < lgNumCores; j++)
core->myFlags[i][j * CacheLineSize] = 0;
core->partnerFlags[i] = (uint8_t **)
_mm_malloc(lgNumCores * sizeof(uint8_t *), 64);
}
core->parity = 0;
core->sense = 1;

cores[cid] = core;
}


if (atomic_dec_and_test(&threadsWaiting)) {
atomic_set(&threadsWaiting, numThreads);
initState = 1;
} else while (initState == 0);


threadCores[tid] = cores[cid];
coreTids[tid] = coreTid;


if (coreTid == 0) {
for (int i = 0; i < lgNumCores; i++) {

int partnerCid = (cid + (1 << i)) % numCores;
for (int j = 0; j < 2; j++)
core->partnerFlags[j][i] = (uint8_t *)
&cores[partnerCid]->myFlags[j][i * CacheLineSize];
}
}


if (atomic_dec_and_test(&threadsWaiting)) {
atomic_set(&threadsWaiting, numThreads);
initState = 2;
} else while (initState == 1);
}



void Barrier::wait(int tid) {
int i, di;

#if (__MIC__)
uint8_t sendbuf[CacheLineSize] __attribute((aligned(CacheLineSize)));
__m512d Vt;
#endif


CoreBarrier *bar = threadCores[tid];
int8_t coreTid = coreTids[tid];


bar->threadSenses[coreTid] = !bar->threadSenses[coreTid];


if (coreTid == 0) {


for (i = 1; i < numThreadsPerCore; i++) {
while (bar->threadSenses[i] == bar->coreSense)
cpu_pause();
}


if (numCores > 1) {
#if (__MIC__)
_mm_prefetch((const char *)bar->partnerFlags[bar->parity][0],
_MM_HINT_ET1);


sendbuf[0] = bar->sense;
Vt = _mm512_load_pd((void *)sendbuf);
#endif


for (i = di = 0; i < lgNumCores - 1; i++, di += CacheLineSize) {

#if (__MIC__)
_mm_prefetch((const char *)bar->partnerFlags[bar->parity][i+1],
_MM_HINT_ET1);


_mm512_storenrngo_pd((void *)bar->partnerFlags[bar->parity][i],
Vt);
#else
*bar->partnerFlags[bar->parity][i] = bar->sense;
#endif


while (bar->myFlags[bar->parity][di] != bar->sense)
cpu_pause();
}
#if (__MIC__)
_mm512_storenrngo_pd((void *)bar->partnerFlags[bar->parity][i], Vt);
#else
*bar->partnerFlags[bar->parity][i] = bar->sense;
#endif
while (bar->myFlags[bar->parity][di] != bar->sense)
cpu_pause();


if (bar->parity == 1) bar->sense = !bar->sense;
bar->parity = 1 - bar->parity;
}


bar->coreSense = bar->threadSenses[0];
}


else {
#if 0
if (coreTid == 1) {
for (i = di = 0;  i < lgNumCores;  i++, di += CacheLineSize)
_mm_prefetch((const char *)bar->partnerFlags[bar->parity][i],
_MM_HINT_ET1);
}
#endif

while (bar->coreSense != bar->threadSenses[coreTid])
cpu_pause();
}
}



Barrier::~Barrier() {
if (initState) {
for (int i = 0; i < numCores; i++) {
_mm_free((void *) cores[i]->threadSenses);
for (int j = 0; j < 2; j++) {
_mm_free((void *) cores[i]->myFlags[j]);
_mm_free((void *) cores[i]->partnerFlags[j]);
}
_mm_free(cores[i]);
}
}

_mm_free(coreTids);
_mm_free(threadCores);
_mm_free(cores);
}

}

