#include <omp.h>
#include <cstdio>
#include <atomic>


#include <cstdint>
#include <atomic>

#define CACHE_ALIGNED alignas(64)

typedef uint32_t unsignedType;
typedef uint64_t pairType;

class contiguousWork {
union CACHE_ALIGNED {
struct {
std::atomic<unsignedType> atomicBase;
std::atomic<unsignedType> atomicEnd;
} ab;
pairType pair;
std::atomic<pairType> atomicPair;
};
auto setBase(unsignedType nb) {
return ab.atomicBase.store(nb, std::memory_order_relaxed);
}
void assign(unsignedType b, unsignedType e) {
ab.atomicBase.store(b, std::memory_order_relaxed);
ab.atomicEnd.store (e, std::memory_order_relaxed);
}

public:
contiguousWork() {}
contiguousWork(unsignedType b, unsignedType e) {
assign(b,e);
}
auto getBase(std::memory_order order = std::memory_order_relaxed) const {
return ab.atomicBase.load(order);
}
auto getEnd(std::memory_order order = std::memory_order_relaxed) const {
return ab.atomicEnd.load(order);
}
contiguousWork(contiguousWork * other) {
assign(other->getBase(), other->getEnd());
}
~contiguousWork() {}

auto getIterations() const {
return getEnd() - getBase();
}
bool trySteal(unsignedType * basep, unsignedType * endp);
bool incrementBase(unsignedType * oldp);
};

bool contiguousWork::incrementBase(unsignedType * basep) {
auto oldBase = getBase();
auto oldEnd  = getEnd();

if (oldBase >= oldEnd) {
return false;
}
setBase(oldBase + 1);
std::atomic_thread_fence(std::memory_order_seq_cst);

auto newEnd = getEnd();

if (newEnd == oldBase) {
return false;
}
else {
*basep = oldBase;
return true;
}
}

bool contiguousWork::trySteal(unsignedType * basep,
unsignedType * endp) {

contiguousWork oldValues(this);
for (;;) {
auto oldBase = oldValues.getBase();
auto oldEnd  = oldValues.getEnd();

if (oldBase >= oldEnd) {
return false;
}
auto newEnd = oldEnd - 1;
contiguousWork newValues(oldBase, newEnd);
if (atomicPair.compare_exchange_weak(oldValues.pair,
newValues.pair,
std::memory_order_acq_rel)) {
*basep = newEnd;
*endp = oldEnd;
return true;
}
}
}

#include <unordered_map>

class lockedHash {
std::unordered_map<unsignedType, int> theMap;
omp_lock_t theLock;

public:
lockedHash() {
omp_init_lock(&theLock);
}
void insert(unsignedType key, int value) {
omp_set_lock(&theLock); 
theMap.insert({key, value});
omp_unset_lock(&theLock); 
}
auto lookup(unsignedType key) {
omp_set_lock(&theLock); 
auto result = theMap.find(key);
omp_unset_lock(&theLock); 
return result == theMap.end() ? -1 : result->second;
}
std::unordered_map<unsignedType, int> & getMap() {
return theMap;
}
};

#if (0)
void doInsert(int me, lockedHash *theMap, unsignedType value) {
auto prev = theMap->lookup(value);
if (prev != -1) {
printf("Iteration %lu executed by %d and %d\n", value, prev, me);
return;
}
theMap->insert(value, me);
}
#else
#  define doInsert(a,b,c) (void)0
#endif

int main(int, char **) {
enum {
iterationsToExecute = 100000
};
contiguousWork theWork(0,iterationsToExecute);
int totalIterations = 0;
lockedHash iterationMap;
auto numThreads = omp_get_max_threads();

#pragma omp parallel reduction(+:totalIterations)
{
auto me = omp_get_thread_num();

if (me == 0) {
#if (1)
unsignedType myIteration;
while (theWork.incrementBase(&myIteration)) {
doInsert(me, &iterationMap, myIteration);
totalIterations++;
}
#endif
} else {
unsignedType myBase, myEnd;
while (theWork.trySteal(&myBase,&myEnd)) {
doInsert(me, &iterationMap, myBase);
totalIterations++;
}
}
}
printf("%d iterations of %d executed\n", totalIterations, iterationsToExecute);
if (iterationsToExecute == totalIterations) {
return 0;
}
return -1;
}
