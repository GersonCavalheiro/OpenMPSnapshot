


#include "tbbmalloc_internal.h"
#include <errno.h>
#include <new>        
#include <string.h>   

#include "../tbb/tbb_version.h"
#include "../tbb/itt_notify.h" 

#if USE_PTHREAD
#define TlsSetValue_func pthread_setspecific
#define TlsGetValue_func pthread_getspecific
#define GetMyTID() pthread_self()
#include <sched.h>
inline void do_yield() {sched_yield();}
extern "C" { static void mallocThreadShutdownNotification(void*); }
#if __sun || __SUNPRO_CC
#define __asm__ asm
#endif
#include <unistd.h> 
#elif USE_WINTHREAD
#define GetMyTID() GetCurrentThreadId()
#if __TBB_WIN8UI_SUPPORT
#include<thread>
#define TlsSetValue_func FlsSetValue
#define TlsGetValue_func FlsGetValue
#define TlsAlloc() FlsAlloc(NULL)
#define TLS_ALLOC_FAILURE FLS_OUT_OF_INDEXES
#define TlsFree FlsFree
inline void do_yield() {std::this_thread::yield();}
#else
#define TlsSetValue_func TlsSetValue
#define TlsGetValue_func TlsGetValue
#define TLS_ALLOC_FAILURE TLS_OUT_OF_INDEXES
inline void do_yield() {SwitchToThread();}
#endif
#else
#error Must define USE_PTHREAD or USE_WINTHREAD

#endif


#define FREELIST_NONBLOCKING 1

namespace rml {
class MemoryPool;
namespace internal {

class Block;
class MemoryPool;

#if MALLOC_CHECK_RECURSION

inline bool isMallocInitialized();

bool RecursiveMallocCallProtector::noRecursion() {
MALLOC_ASSERT(isMallocInitialized(),
"Recursion status can be checked only when initialization was done.");
return !mallocRecursionDetected;
}

#endif 


const uint16_t startupAllocObjSizeMark = ~(uint16_t)0;


const int blockHeaderAlignment = estimatedCacheLineSize;




class BootStrapBlocks {
MallocMutex bootStrapLock;
Block      *bootStrapBlock;
Block      *bootStrapBlockUsed;
FreeObject *bootStrapObjectList;
public:
void *allocate(MemoryPool *memPool, size_t size);
void free(void* ptr);
void reset();
};

#if USE_INTERNAL_TID
class ThreadId {
static tls_key_t Tid_key;
static intptr_t ThreadCount;

unsigned int id;

static unsigned int tlsNumber() {
unsigned int result = reinterpret_cast<intptr_t>(TlsGetValue_func(Tid_key));
if( !result ) {
RecursiveMallocCallProtector scoped;
result = AtomicIncrement(ThreadCount); 
TlsSetValue_func( Tid_key, reinterpret_cast<void*>(result) );
}
return result;
}
public:
static void init() {
#if USE_WINTHREAD
Tid_key = TlsAlloc();
#else
int status = pthread_key_create( &Tid_key, NULL );
if ( status ) {
fprintf (stderr, "The memory manager cannot create tls key during initialization; exiting \n");
exit(1);
}
#endif 
}
static void destroy() {
if( Tid_key ) {
#if USE_WINTHREAD
BOOL status = !(TlsFree( Tid_key ));  
#else
int status = pthread_key_delete( Tid_key );
#endif 
if ( status ) {
fprintf (stderr, "The memory manager cannot delete tls key; exiting \n");
exit(1);
}
Tid_key = 0;
}
}

ThreadId() : id(ThreadId::tlsNumber()) {}
bool isCurrentThreadId() const { return id == ThreadId::tlsNumber(); }

#if COLLECT_STATISTICS || MALLOC_TRACE
friend unsigned int getThreadId() { return ThreadId::tlsNumber(); }
#endif
#if COLLECT_STATISTICS
static unsigned getMaxThreadId() { return ThreadCount; }

friend int STAT_increment(ThreadId tid, int bin, int ctr);
#endif
};

tls_key_t ThreadId::Tid_key;
intptr_t ThreadId::ThreadCount;

#if COLLECT_STATISTICS
int STAT_increment(ThreadId tid, int bin, int ctr)
{
return ::STAT_increment(tid.id, bin, ctr);
}
#endif

#else 

class ThreadId {
#if USE_PTHREAD
pthread_t tid;
#else
DWORD     tid;
#endif
public:
ThreadId() : tid(GetMyTID()) {}
#if USE_PTHREAD
bool isCurrentThreadId() const { return pthread_equal(pthread_self(), tid); }
#else
bool isCurrentThreadId() const { return GetCurrentThreadId() == tid; }
#endif
static void init() {}
static void destroy() {}
};

#endif 



bool TLSKey::init()
{
#if USE_WINTHREAD
TLS_pointer_key = TlsAlloc();
if (TLS_pointer_key == TLS_ALLOC_FAILURE)
return false;
#else
int status = pthread_key_create( &TLS_pointer_key, mallocThreadShutdownNotification );
if ( status )
return false;
#endif 
return true;
}

bool TLSKey::destroy()
{
#if USE_WINTHREAD
BOOL status1 = !(TlsFree(TLS_pointer_key)); 
#else
int status1 = pthread_key_delete(TLS_pointer_key);
#endif 
MALLOC_ASSERT(!status1, "The memory manager cannot delete tls key.");
return status1==0;
}

inline TLSData* TLSKey::getThreadMallocTLS() const
{
return (TLSData *)TlsGetValue_func( TLS_pointer_key );
}

inline void TLSKey::setThreadMallocTLS( TLSData * newvalue ) {
RecursiveMallocCallProtector scoped;
TlsSetValue_func( TLS_pointer_key, newvalue );
}


class Bin;
class StartupBlock;

class MemoryPool {
static const size_t defaultGranularity = estimatedCacheLineSize;

MemoryPool();                  
public:
static MallocMutex  memPoolListLock;

MemoryPool    *next,
*prev;
ExtMemoryPool  extMemPool;
BootStrapBlocks bootStrapBlocks;

bool init(intptr_t poolId, const MemPoolPolicy* memPoolPolicy);
static void initDefaultPool();
bool reset();
bool destroy();
void processThreadShutdown(TLSData *tlsData);

inline TLSData *getTLS(bool create);
void clearTLS() { extMemPool.tlsPointerKey.setThreadMallocTLS(NULL); }

Block *getEmptyBlock(size_t size);
void returnEmptyBlock(Block *block, bool poolTheBlock);

void *getFromLLOCache(TLSData *tls, size_t size, size_t alignment);
void putToLLOCache(TLSData *tls, void *object);
};

static intptr_t defaultMemPool_space[sizeof(MemoryPool)/sizeof(intptr_t) +
(sizeof(MemoryPool)%sizeof(intptr_t)? 1 : 0)];
static MemoryPool *defaultMemPool = (MemoryPool*)defaultMemPool_space;
const size_t MemoryPool::defaultGranularity;
MallocMutex  MemoryPool::memPoolListLock;
HugePagesStatus hugePages;
static bool usedBySrcIncluded = false;

template<size_t padd>
struct PaddingImpl {
size_t       __padding[padd];
};

template<>
struct PaddingImpl<0> {};

template<int N>
struct Padding : PaddingImpl<N/sizeof(size_t)> {};

class GlobalBlockFields : public BlockI {
protected:
FreeObject  *publicFreeList;
Block       *nextPrivatizable;
MemoryPool  *poolPtr;
};

class LocalBlockFields : public GlobalBlockFields, Padding<blockHeaderAlignment - sizeof(GlobalBlockFields)>  {
protected:
Block       *next;
Block       *previous;        
FreeObject  *bumpPtr;         
FreeObject  *freeList;

TLSData     *tlsPtr;
ThreadId     ownerTid;        
BackRefIdx   backRefIdx;
uint16_t     allocatedCount;  
uint16_t     objectSize;
bool         isFull;

friend class FreeBlockPool;
friend class StartupBlock;
friend class LifoList;
friend void *BootStrapBlocks::allocate(MemoryPool *, size_t);
friend bool OrphanedBlocks::cleanup(Backend*);
friend Block *MemoryPool::getEmptyBlock(size_t);
};

class Block : public LocalBlockFields,
Padding<2*blockHeaderAlignment - sizeof(LocalBlockFields)> {
public:
bool empty() const { return allocatedCount==0 && publicFreeList==NULL; }
inline FreeObject* allocate();
inline FreeObject *allocateFromFreeList();
inline bool emptyEnoughToUse();
bool freeListNonNull() { return freeList; }
void freePublicObject(FreeObject *objectToFree);
inline void freeOwnObject(void *object);
void makeEmpty();
void privatizePublicFreeList( bool cleanup = false );
void restoreBumpPtr();
void privatizeOrphaned(TLSData *tls, unsigned index);
void shareOrphaned(intptr_t binTag, unsigned index);
unsigned int getSize() const {
MALLOC_ASSERT(isStartupAllocObject() || objectSize<minLargeObjectSize,
"Invalid object size");
return isStartupAllocObject()? 0 : objectSize;
}
const BackRefIdx *getBackRefIdx() const { return &backRefIdx; }
inline bool isOwnedByCurrentThread() const;
bool isStartupAllocObject() const { return objectSize == startupAllocObjSizeMark; }
inline FreeObject *findObjectToFree(const void *object) const;
void checkFreePrecond(const void *object) const {
#if MALLOC_DEBUG
const char *msg = "Possible double free or heap corruption.";
MALLOC_ASSERT(isAligned(object, sizeof(size_t)), "Try to free invalid small object");
MALLOC_ASSERT(allocatedCount>0, msg);
MALLOC_ASSERT((uintptr_t)object - (uintptr_t)this >= sizeof(Block), msg);
if (startupAllocObjSizeMark == objectSize) 
MALLOC_ASSERT(object<=bumpPtr, msg);
else {
MALLOC_ASSERT(isAligned(object, 8), "Try to free invalid small object");
MALLOC_ASSERT(allocatedCount <= (slabSize-sizeof(Block))/objectSize
&& (!bumpPtr || object>bumpPtr), msg);
FreeObject *toFree = findObjectToFree(object);
MALLOC_ASSERT(toFree != freeList, msg);
MALLOC_ASSERT(toFree != publicFreeList, msg);
}
#else
suppress_unused_warning(object);
#endif
}
void initEmptyBlock(TLSData *tls, size_t size);
size_t findObjectSize(void *object) const;
MemoryPool *getMemPool() const { return poolPtr; } 

protected:
void cleanBlockHeader();

private:
static const float emptyEnoughRatio; 

inline FreeObject *allocateFromBumpPtr();
inline FreeObject *findAllocatedObject(const void *address) const;
inline bool isProperlyPlaced(const void *object) const;
inline void markOwned(TLSData *tls) {
MALLOC_ASSERT(!tlsPtr, ASSERT_TEXT);
ownerTid = ThreadId(); 
tlsPtr = tls;
}
inline void markOrphaned() {
MALLOC_ASSERT(tlsPtr, ASSERT_TEXT);
tlsPtr = NULL;
}

friend class Bin;
friend class TLSData;
friend bool MemoryPool::destroy();
};

const float Block::emptyEnoughRatio = 1.0 / 4.0;

MALLOC_STATIC_ASSERT(sizeof(Block) <= 2*estimatedCacheLineSize,
"The class Block does not fit into 2 cache lines on this platform. "
"Defining USE_INTERNAL_TID may help to fix it.");

class Bin {
Block      *activeBlk;
Block      *mailbox;
MallocMutex mailLock;

public:
inline Block* getActiveBlock() const { return activeBlk; }
void resetActiveBlock() { activeBlk = 0; }
bool activeBlockUnused() const { return activeBlk && !activeBlk->allocatedCount; }
inline void setActiveBlock(Block *block);
inline Block* setPreviousBlockActive();
Block* getPublicFreeListBlock();
void moveBlockToFront(Block *block);
void processLessUsedBlock(MemoryPool *memPool, Block *block);

void outofTLSBin(Block* block);
void verifyTLSBin(size_t size) const;
void pushTLSBin(Block* block);

void verifyInitState() const {
MALLOC_ASSERT( activeBlk == 0, ASSERT_TEXT );
MALLOC_ASSERT( mailbox == 0, ASSERT_TEXT );
}

friend void Block::freePublicObject (FreeObject *objectToFree);
};




const uint32_t minSmallObjectIndex = 0;
const uint32_t numSmallObjectBins = 8;
const uint32_t maxSmallObjectSize = 64;


const uint32_t minSegregatedObjectIndex = minSmallObjectIndex+numSmallObjectBins;
const uint32_t numSegregatedObjectBins = 16;
const uint32_t maxSegregatedObjectSize = 1024;


const uint32_t minFittingIndex = minSegregatedObjectIndex+numSegregatedObjectBins;
const uint32_t numFittingBins = 5;

const uint32_t fittingAlignment = estimatedCacheLineSize;

#define SET_FITTING_SIZE(N) ( (slabSize-sizeof(Block))/N ) & ~(fittingAlignment-1)
const uint32_t fittingSize1 = SET_FITTING_SIZE(9); 
const uint32_t fittingSize2 = SET_FITTING_SIZE(6); 
const uint32_t fittingSize3 = SET_FITTING_SIZE(4); 
const uint32_t fittingSize4 = SET_FITTING_SIZE(3); 
const uint32_t fittingSize5 = SET_FITTING_SIZE(2); 
#undef SET_FITTING_SIZE


const uint32_t numBlockBins = minFittingIndex+numFittingBins;


const uint32_t minLargeObjectSize = fittingSize5 + 1;


class FreeBlockPool {
Block      *head;
int         size;
Backend    *backend;
bool        lastAccessMiss;
public:
static const int POOL_HIGH_MARK = 32;
static const int POOL_LOW_MARK  = 8;

class ResOfGet {
ResOfGet();
public:
Block* block;
bool   lastAccMiss;
ResOfGet(Block *b, bool lastMiss) : block(b), lastAccMiss(lastMiss) {}
};

FreeBlockPool(Backend *bknd) : backend(bknd) {}
ResOfGet getBlock();
void returnBlock(Block *block);
bool externalCleanup(); 
};

template<int LOW_MARK, int HIGH_MARK>
class LocalLOCImpl {
static const size_t MAX_TOTAL_SIZE = 4*1024*1024;
LargeMemoryBlock *head,
*tail; 
size_t            totalSize;
int               numOfBlocks;
public:
bool put(LargeMemoryBlock *object, ExtMemoryPool *extMemPool);
LargeMemoryBlock *get(size_t size);
bool externalCleanup(ExtMemoryPool *extMemPool);
#if __TBB_MALLOC_WHITEBOX_TEST
LocalLOCImpl() : head(NULL), tail(NULL), totalSize(0), numOfBlocks(0) {}
static size_t getMaxSize() { return MAX_TOTAL_SIZE; }
static const int LOC_HIGH_MARK = HIGH_MARK;
#else
#endif
};

typedef LocalLOCImpl<8,32> LocalLOC; 

class TLSData : public TLSRemote {
MemoryPool   *memPool;
public:
Bin           bin[numBlockBinLimit];
FreeBlockPool freeSlabBlocks;
LocalLOC      lloc;
unsigned      currCacheIdx;
private:
bool unused;
public:
TLSData(MemoryPool *mPool, Backend *bknd) : memPool(mPool), freeSlabBlocks(bknd) {}
MemoryPool *getMemPool() const { return memPool; }
Bin* getAllocationBin(size_t size);
void release(MemoryPool *mPool);
bool externalCleanup(ExtMemoryPool *mPool, bool cleanOnlyUnused) {
if (!unused && cleanOnlyUnused) return false;
return lloc.externalCleanup(mPool) | freeSlabBlocks.externalCleanup();
}
bool cleanUnusedActiveBlocks(Backend *backend, bool userPool);
void markUsed() { unused = false; } 
void markUnused() { unused =  true; } 
};

TLSData *TLSKey::createTLS(MemoryPool *memPool, Backend *backend)
{
MALLOC_ASSERT( sizeof(TLSData) >= sizeof(Bin) * numBlockBins + sizeof(FreeBlockPool), ASSERT_TEXT );
TLSData* tls = (TLSData*) memPool->bootStrapBlocks.allocate(memPool, sizeof(TLSData));
if ( !tls )
return NULL;
new(tls) TLSData(memPool, backend);

#if MALLOC_DEBUG
for (uint32_t i = 0; i < numBlockBinLimit; i++)
tls->bin[i].verifyInitState();
#endif
setThreadMallocTLS(tls);
memPool->extMemPool.allLocalCaches.registerThread(tls);
return tls;
}

bool TLSData::cleanUnusedActiveBlocks(Backend *backend, bool userPool)
{
bool released = false;
for (uint32_t i=0; i<numBlockBinLimit; i++)
if (bin[i].activeBlockUnused()) {
Block *block = bin[i].getActiveBlock();
bin[i].outofTLSBin(block);
if (!userPool)
removeBackRef(*(block->getBackRefIdx()));
backend->putSlabBlock(block);

released = true;
}
return released;
}

bool ExtMemoryPool::releaseAllLocalCaches()
{
bool released = allLocalCaches.cleanup(this, false);

if (TLSData *tlsData = tlsPointerKey.getThreadMallocTLS())
released |= tlsData->cleanUnusedActiveBlocks(&backend, userPool());

return released;
}

void AllLocalCaches::registerThread(TLSRemote *tls)
{
tls->prev = NULL;
MallocMutex::scoped_lock lock(listLock);
MALLOC_ASSERT(head!=tls, ASSERT_TEXT);
tls->next = head;
if (head)
head->prev = tls;
head = tls;
MALLOC_ASSERT(head->next!=head, ASSERT_TEXT);
}

void AllLocalCaches::unregisterThread(TLSRemote *tls)
{
MallocMutex::scoped_lock lock(listLock);
MALLOC_ASSERT(head, "Can't unregister thread: no threads are registered.");
if (head == tls)
head = tls->next;
if (tls->next)
tls->next->prev = tls->prev;
if (tls->prev)
tls->prev->next = tls->next;
MALLOC_ASSERT(!tls->next || tls->next->next!=tls->next, ASSERT_TEXT);
}

bool AllLocalCaches::cleanup(ExtMemoryPool *extPool, bool cleanOnlyUnused)
{
bool total = false;
{
MallocMutex::scoped_lock lock(listLock);

for (TLSRemote *curr=head; curr; curr=curr->next)
total |= static_cast<TLSData*>(curr)->
externalCleanup(extPool, cleanOnlyUnused);
}
return total;
}

void AllLocalCaches::markUnused()
{
bool locked;
MallocMutex::scoped_lock lock(listLock, false, &locked);
if (!locked) 
return;

for (TLSRemote *curr=head; curr; curr=curr->next)
static_cast<TLSData*>(curr)->markUnused();
}

#if MALLOC_CHECK_RECURSION
MallocMutex RecursiveMallocCallProtector::rmc_mutex;
pthread_t   RecursiveMallocCallProtector::owner_thread;
void       *RecursiveMallocCallProtector::autoObjPtr;
bool        RecursiveMallocCallProtector::mallocRecursionDetected;
#if __FreeBSD__
bool        RecursiveMallocCallProtector::canUsePthread;
#endif

#endif



enum MemoryOrigin {
ourMem,    
unknownMem 
};

template<MemoryOrigin> bool isLargeObject(void *object);
static void *internalMalloc(size_t size);
static void internalFree(void *object);
static void *internalPoolMalloc(MemoryPool* mPool, size_t size);
static bool internalPoolFree(MemoryPool *mPool, void *object, size_t size);

#if !MALLOC_DEBUG
#if __INTEL_COMPILER || _MSC_VER
#define NOINLINE(decl) __declspec(noinline) decl
#define ALWAYSINLINE(decl) __forceinline decl
#elif __GNUC__
#define NOINLINE(decl) decl __attribute__ ((noinline))
#define ALWAYSINLINE(decl) decl __attribute__ ((always_inline))
#else
#define NOINLINE(decl) decl
#define ALWAYSINLINE(decl) decl
#endif

static NOINLINE( bool doInitialization() );
ALWAYSINLINE( bool isMallocInitialized() );

#undef ALWAYSINLINE
#undef NOINLINE
#endif 





#if _WIN64 && _MSC_VER>=1400 && !__INTEL_COMPILER
extern "C" unsigned char _BitScanReverse( unsigned long* i, unsigned long w );
#pragma intrinsic(_BitScanReverse)
#endif
static inline unsigned int highestBitPos(unsigned int n)
{
MALLOC_ASSERT( n>=64 && n<1024, ASSERT_TEXT ); 
unsigned int pos;
#if __ARCH_x86_32||__ARCH_x86_64

# if __linux__||__APPLE__||__FreeBSD__||__NetBSD__||__sun||__MINGW32__
__asm__ ("bsr %1,%0" : "=r"(pos) : "r"(n));
# elif (_WIN32 && (!_WIN64 || __INTEL_COMPILER))
__asm
{
bsr eax, n
mov pos, eax
}
# elif _WIN64 && _MSC_VER>=1400
_BitScanReverse((unsigned long*)&pos, (unsigned long)n);
# else
#   error highestBitPos() not implemented for this platform
# endif
#elif __arm__
__asm__ __volatile__
(
"clz %0, %1\n"
"rsb %0, %0, %2\n"
:"=r" (pos) :"r" (n), "I" (31)
);
#else
static unsigned int bsr[16] = {0,6,7,7,8,8,8,8,9,9,9,9,9,9,9,9};
pos = bsr[ n>>6 ];
#endif 
return pos;
}

template<bool Is32Bit>
unsigned int getSmallObjectIndex(unsigned int size)
{
return (size-1)>>3;
}
template<>
unsigned int getSmallObjectIndex<false>(unsigned int size)
{
unsigned int result = (size-1)>>3;
if (result) result |= 1; 
return result;
}

template<bool indexRequest>
static unsigned int getIndexOrObjectSize (unsigned int size)
{
if (size <= maxSmallObjectSize) { 
unsigned int index = getSmallObjectIndex<(sizeof(size_t)<=4)>( size );

return indexRequest ? index : (index+1)<<3;
}
else if (size <= maxSegregatedObjectSize ) { 
unsigned int order = highestBitPos(size-1); 
MALLOC_ASSERT( 6<=order && order<=9, ASSERT_TEXT );
if (indexRequest)
return minSegregatedObjectIndex - (4*6) - 4 + (4*order) + ((size-1)>>(order-2));
else {
unsigned int alignment = 128 >> (9-order); 
MALLOC_ASSERT( alignment==16 || alignment==32 || alignment==64 || alignment==128, ASSERT_TEXT );
return alignUp(size,alignment);
}
}
else {
if( size <= fittingSize3 ) {
if( size <= fittingSize2 ) {
if( size <= fittingSize1 )
return indexRequest ? minFittingIndex : fittingSize1;
else
return indexRequest ? minFittingIndex+1 : fittingSize2;
} else
return indexRequest ? minFittingIndex+2 : fittingSize3;
} else {
if( size <= fittingSize5 ) {
if( size <= fittingSize4 )
return indexRequest ? minFittingIndex+3 : fittingSize4;
else
return indexRequest ? minFittingIndex+4 : fittingSize5;
} else {
MALLOC_ASSERT( 0,ASSERT_TEXT ); 
return ~0U;
}
}
}
}

static unsigned int getIndex (unsigned int size)
{
return getIndexOrObjectSize<true>(size);
}

static unsigned int getObjectSize (unsigned int size)
{
return getIndexOrObjectSize<false>(size);
}


void *BootStrapBlocks::allocate(MemoryPool *memPool, size_t size)
{
FreeObject *result;

MALLOC_ASSERT( size == sizeof(TLSData), ASSERT_TEXT );

{ 
MallocMutex::scoped_lock scoped_cs(bootStrapLock);

if( bootStrapObjectList) {
result = bootStrapObjectList;
bootStrapObjectList = bootStrapObjectList->next;
} else {
if (!bootStrapBlock) {
bootStrapBlock = memPool->getEmptyBlock(size);
if (!bootStrapBlock) return NULL;
}
result = bootStrapBlock->bumpPtr;
bootStrapBlock->bumpPtr = (FreeObject *)((uintptr_t)bootStrapBlock->bumpPtr - bootStrapBlock->objectSize);
if ((uintptr_t)bootStrapBlock->bumpPtr < (uintptr_t)bootStrapBlock+sizeof(Block)) {
bootStrapBlock->bumpPtr = NULL;
bootStrapBlock->next = bootStrapBlockUsed;
bootStrapBlockUsed = bootStrapBlock;
bootStrapBlock = NULL;
}
}
} 

memset (result, 0, size);
return (void*)result;
}

void BootStrapBlocks::free(void* ptr)
{
MALLOC_ASSERT( ptr, ASSERT_TEXT );
{ 
MallocMutex::scoped_lock scoped_cs(bootStrapLock);
((FreeObject*)ptr)->next = bootStrapObjectList;
bootStrapObjectList = (FreeObject*)ptr;
} 
}

void BootStrapBlocks::reset()
{
bootStrapBlock = bootStrapBlockUsed = NULL;
bootStrapObjectList = NULL;
}

#if !(FREELIST_NONBLOCKING)
static MallocMutex publicFreeListLock; 
#endif

const uintptr_t UNUSABLE = 0x1;
inline bool isSolidPtr( void* ptr )
{
return (UNUSABLE|(uintptr_t)ptr)!=UNUSABLE;
}
inline bool isNotForUse( void* ptr )
{
return (uintptr_t)ptr==UNUSABLE;
}




LifoList::LifoList( ) : top(NULL)
{
memset(&lock, 0, sizeof(MallocMutex));
}

void LifoList::push(Block *block)
{
MallocMutex::scoped_lock scoped_cs(lock);
block->next = top;
top = block;
}

Block *LifoList::pop()
{
Block *block=NULL;
if (top) {
MallocMutex::scoped_lock scoped_cs(lock);
if (top) {
block = top;
top = block->next;
}
}
return block;
}

Block *LifoList::grab()
{
Block *block = NULL;
if (top) {
MallocMutex::scoped_lock scoped_cs(lock);
block = top;
top = NULL;
}
return block;
}



template<bool poolDestroy> void AllLargeBlocksList::releaseAll(Backend *backend) {
LargeMemoryBlock *next, *lmb = loHead;
loHead = NULL;

for (; lmb; lmb = next) {
next = lmb->gNext;
if (poolDestroy) {
removeBackRef(lmb->backRefIdx);
} else {
lmb->gNext = lmb->gPrev = NULL;
backend->returnLargeObject(lmb);
}
}
}

TLSData* MemoryPool::getTLS(bool create)
{
TLSData* tls = extMemPool.tlsPointerKey.getThreadMallocTLS();
if (create && !tls)
tls = extMemPool.tlsPointerKey.createTLS(this, &extMemPool.backend);
return tls;
}


inline Bin* TLSData::getAllocationBin(size_t size)
{
return bin + getIndex(size);
}


Block *MemoryPool::getEmptyBlock(size_t size)
{
TLSData* tls = extMemPool.tlsPointerKey.getThreadMallocTLS();
FreeBlockPool::ResOfGet resOfGet = tls?
tls->freeSlabBlocks.getBlock() : FreeBlockPool::ResOfGet(NULL, false);
Block *result = resOfGet.block;

if (!result) { 
int num = resOfGet.lastAccMiss? Backend::numOfSlabAllocOnMiss : 1;
BackRefIdx backRefIdx[Backend::numOfSlabAllocOnMiss];

result = static_cast<Block*>(extMemPool.backend.getSlabBlock(num));
if (!result) return NULL;

if (!extMemPool.userPool())
for (int i=0; i<num; i++) {
backRefIdx[i] = BackRefIdx::newBackRef(false);
if (backRefIdx[i].isInvalid()) {
for (int j=0; j<i; j++)
removeBackRef(backRefIdx[j]);
Block *b = result;
for (int j=0; j<num; b=(Block*)((uintptr_t)b+slabSize), j++)
extMemPool.backend.putSlabBlock(b);
return NULL;
}
}
Block *b = result;
for (int i=0; i<num; b=(Block*)((uintptr_t)b+slabSize), i++) {
if (extMemPool.userPool()) {
new (&b->backRefIdx) BackRefIdx();
} else {
setBackRef(backRefIdx[i], b);
b->backRefIdx = backRefIdx[i];
}
b->tlsPtr = tls;
b->poolPtr = this;
if (i > 0) {
MALLOC_ASSERT(tls, ASSERT_TEXT);
tls->freeSlabBlocks.returnBlock(b);
}
}
}
MALLOC_ASSERT(result, ASSERT_TEXT);
result->initEmptyBlock(tls, size);
STAT_increment(getThreadId(), getIndex(result->objectSize), allocBlockNew);
return result;
}

void MemoryPool::returnEmptyBlock(Block *block, bool poolTheBlock)
{
block->makeEmpty();
if (poolTheBlock) {
extMemPool.tlsPointerKey.getThreadMallocTLS()->freeSlabBlocks.returnBlock(block);
}
else {
if (!extMemPool.userPool())
removeBackRef(*(block->getBackRefIdx()));
extMemPool.backend.putSlabBlock(block);
}
}

bool ExtMemoryPool::init(intptr_t poolId, rawAllocType rawAlloc,
rawFreeType rawFree, size_t granularity,
bool keepAllMemory, bool fixedPool)
{
this->poolId = poolId;
this->rawAlloc = rawAlloc;
this->rawFree = rawFree;
this->granularity = granularity;
this->keepAllMemory = keepAllMemory;
this->fixedPool = fixedPool;
this->delayRegsReleasing = false;
if (! initTLS())
return false;
loc.init(this);
backend.init(this);
MALLOC_ASSERT(isPoolValid(), NULL);
return true;
}

bool ExtMemoryPool::initTLS() { return tlsPointerKey.init(); }

bool MemoryPool::init(intptr_t poolId, const MemPoolPolicy *policy)
{
if (!extMemPool.init(poolId, policy->pAlloc, policy->pFree,
policy->granularity? policy->granularity : defaultGranularity,
policy->keepAllMemory, policy->fixedPool))
return false;
{
MallocMutex::scoped_lock lock(memPoolListLock);
next = defaultMemPool->next;
defaultMemPool->next = this;
prev = defaultMemPool;
if (next)
next->prev = this;
}
return true;
}

bool MemoryPool::reset()
{
MALLOC_ASSERT(extMemPool.userPool(), "No reset for the system pool.");
extMemPool.delayRegionsReleasing(true);

bootStrapBlocks.reset();
extMemPool.lmbList.releaseAll<false>(&extMemPool.backend);
if (!extMemPool.reset())
return false;

if (!extMemPool.initTLS())
return false;
extMemPool.delayRegionsReleasing(false);
return true;
}

bool MemoryPool::destroy()
{
#if __TBB_MALLOC_LOCACHE_STAT
extMemPool.loc.reportStat(stdout);
#endif
#if __TBB_MALLOC_BACKEND_STAT
extMemPool.backend.reportStat(stdout);
#endif
{
MallocMutex::scoped_lock lock(memPoolListLock);
if (prev)
prev->next = next;
if (next)
next->prev = prev;
}
if (extMemPool.userPool())
extMemPool.lmbList.releaseAll<true>(&extMemPool.backend);
else {
MALLOC_ASSERT(this==defaultMemPool, NULL);
bootStrapBlocks.reset();
extMemPool.orphanedBlocks.reset();
}
return extMemPool.destroy();
}

void MemoryPool::processThreadShutdown(TLSData *tlsData)
{
tlsData->release(this);
bootStrapBlocks.free(tlsData);
clearTLS();
}

#if MALLOC_DEBUG
void Bin::verifyTLSBin (size_t size) const
{

uint32_t objSize = getObjectSize(size);

if (activeBlk) {
MALLOC_ASSERT( activeBlk->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( activeBlk->objectSize == objSize, ASSERT_TEXT );
#if MALLOC_DEBUG>1
for (Block* temp = activeBlk->next; temp; temp=temp->next) {
MALLOC_ASSERT( temp!=activeBlk, ASSERT_TEXT );
MALLOC_ASSERT( temp->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( temp->objectSize == objSize, ASSERT_TEXT );
MALLOC_ASSERT( temp->previous->next == temp, ASSERT_TEXT );
if (temp->next) {
MALLOC_ASSERT( temp->next->previous == temp, ASSERT_TEXT );
}
}
for (Block* temp = activeBlk->previous; temp; temp=temp->previous) {
MALLOC_ASSERT( temp!=activeBlk, ASSERT_TEXT );
MALLOC_ASSERT( temp->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( temp->objectSize == objSize, ASSERT_TEXT );
MALLOC_ASSERT( temp->next->previous == temp, ASSERT_TEXT );
if (temp->previous) {
MALLOC_ASSERT( temp->previous->next == temp, ASSERT_TEXT );
}
}
#endif 
}
}
#else 
inline void Bin::verifyTLSBin (size_t) const { }
#endif 


void Bin::pushTLSBin(Block* block)
{

unsigned int size = block->objectSize;

MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( block->objectSize != 0, ASSERT_TEXT );
MALLOC_ASSERT( block->next == NULL, ASSERT_TEXT );
MALLOC_ASSERT( block->previous == NULL, ASSERT_TEXT );

MALLOC_ASSERT( this, ASSERT_TEXT );
verifyTLSBin(size);

block->next = activeBlk;
if( activeBlk ) {
block->previous = activeBlk->previous;
activeBlk->previous = block;
if( block->previous )
block->previous->next = block;
} else {
activeBlk = block;
}

verifyTLSBin(size);
}


void Bin::outofTLSBin(Block* block)
{
unsigned int size = block->objectSize;

MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( block->objectSize != 0, ASSERT_TEXT );

MALLOC_ASSERT( this, ASSERT_TEXT );
verifyTLSBin(size);

if (block == activeBlk) {
activeBlk = block->previous? block->previous : block->next;
}

if (block->previous) {
MALLOC_ASSERT( block->previous->next == block, ASSERT_TEXT );
block->previous->next = block->next;
}
if (block->next) {
MALLOC_ASSERT( block->next->previous == block, ASSERT_TEXT );
block->next->previous = block->previous;
}
block->next = NULL;
block->previous = NULL;

verifyTLSBin(size);
}

Block* Bin::getPublicFreeListBlock()
{
Block* block;
MALLOC_ASSERT( this, ASSERT_TEXT );
MALLOC_ASSERT( !activeBlk && !mailbox || activeBlk && activeBlk->isFull, ASSERT_TEXT );

if (!FencedLoad((intptr_t&)mailbox)) 
return NULL;
else { 
MallocMutex::scoped_lock scoped_cs(mailLock);
block = mailbox;
if( block ) {
MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
MALLOC_ASSERT( !isNotForUse(block->nextPrivatizable), ASSERT_TEXT );
mailbox = block->nextPrivatizable;
block->nextPrivatizable = (Block*) this;
}
}
if( block ) {
MALLOC_ASSERT( isSolidPtr(block->publicFreeList), ASSERT_TEXT );
block->privatizePublicFreeList();
}
return block;
}

bool Block::emptyEnoughToUse()
{
const float threshold = (slabSize - sizeof(Block)) * (1-emptyEnoughRatio);

if (bumpPtr) {

STAT_increment(getThreadId(), getIndex(objectSize), examineEmptyEnough);
isFull = false;
return 1;
}


isFull = (allocatedCount*objectSize > threshold)? true: false;
#if COLLECT_STATISTICS
if (isFull)
STAT_increment(getThreadId(), getIndex(objectSize), examineNotEmpty);
else
STAT_increment(getThreadId(), getIndex(objectSize), examineEmptyEnough);
#endif
return !isFull;
}


void Block::restoreBumpPtr()
{
MALLOC_ASSERT( allocatedCount == 0, ASSERT_TEXT );
MALLOC_ASSERT( publicFreeList == NULL, ASSERT_TEXT );
STAT_increment(getThreadId(), getIndex(objectSize), freeRestoreBumpPtr);
bumpPtr = (FreeObject *)((uintptr_t)this + slabSize - objectSize);
freeList = NULL;
isFull = 0;
}

void Block::freeOwnObject(void *object)
{
tlsPtr->markUsed();
allocatedCount--;
MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
#if COLLECT_STATISTICS
if (tlsPtr->getAllocationBin(objectSize)->getActiveBlock() != this)
STAT_increment(getThreadId(), getIndex(objectSize), freeToInactiveBlock);
else
STAT_increment(getThreadId(), getIndex(objectSize), freeToActiveBlock);
#endif
if (allocatedCount==0 && publicFreeList==NULL) {

MALLOC_ASSERT(!isFull, ASSERT_TEXT);
tlsPtr->getAllocationBin(objectSize)->processLessUsedBlock(poolPtr, this);
} else {
FreeObject *objectToFree = findObjectToFree(object);
objectToFree->next = freeList;
freeList = objectToFree;

if (isFull && emptyEnoughToUse())
tlsPtr->getAllocationBin(objectSize)->moveBlockToFront(this);
}
}

void Block::freePublicObject (FreeObject *objectToFree)
{
FreeObject *localPublicFreeList;

MALLOC_ITT_SYNC_RELEASING(&publicFreeList);
#if FREELIST_NONBLOCKING
FreeObject *temp = publicFreeList;
do {
localPublicFreeList = objectToFree->next = temp;
temp = (FreeObject*)AtomicCompareExchange(
(intptr_t&)publicFreeList,
(intptr_t)objectToFree, (intptr_t)localPublicFreeList );
} while( temp != localPublicFreeList );
#else
STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
{
MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
localPublicFreeList = objectToFree->next = publicFreeList;
publicFreeList = objectToFree;
}
#endif

if( localPublicFreeList==NULL ) {
if( !isNotForUse(nextPrivatizable) ) {
MALLOC_ASSERT( nextPrivatizable!=NULL, ASSERT_TEXT );
Bin* theBin = (Bin*) nextPrivatizable;
MallocMutex::scoped_lock scoped_cs(theBin->mailLock);
nextPrivatizable = theBin->mailbox;
theBin->mailbox = this;
}
}
STAT_increment(getThreadId(), ThreadCommonCounters, freeToOtherThread);
STAT_increment(ownerTid, getIndex(objectSize), freeByOtherThread);
}

void Block::privatizePublicFreeList( bool cleanup )
{
FreeObject *temp, *localPublicFreeList;

MALLOC_ASSERT( cleanup || isOwnedByCurrentThread(), ASSERT_TEXT );
#if FREELIST_NONBLOCKING
temp = publicFreeList;
do {
localPublicFreeList = temp;
temp = (FreeObject*)AtomicCompareExchange(
(intptr_t&)publicFreeList,
0, (intptr_t)localPublicFreeList);
} while( temp != localPublicFreeList );
#else
STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
{
MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
localPublicFreeList = publicFreeList;
publicFreeList = NULL;
}
temp = localPublicFreeList;
#endif
MALLOC_ITT_SYNC_ACQUIRED(&publicFreeList);

MALLOC_ASSERT( cleanup || localPublicFreeList, ASSERT_TEXT );
MALLOC_ASSERT( localPublicFreeList==temp, ASSERT_TEXT );
if( isSolidPtr(temp) ) { 
MALLOC_ASSERT( allocatedCount <= (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );

allocatedCount--;
while( isSolidPtr(temp->next) ){ 
temp = temp->next;
allocatedCount--;
MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
}

temp->next = freeList;
freeList = localPublicFreeList;
STAT_increment(getThreadId(), getIndex(objectSize), allocPrivatized);
}
}

void Block::privatizeOrphaned(TLSData *tls, unsigned index)
{
Bin* bin = tls->bin + index;
STAT_increment(getThreadId(), index, allocBlockPublic);
next = NULL;
previous = NULL;
MALLOC_ASSERT( publicFreeList!=NULL, ASSERT_TEXT );

markOwned(tls);
MALLOC_ASSERT( isNotForUse(nextPrivatizable), ASSERT_TEXT );
nextPrivatizable = (Block*)bin;
privatizePublicFreeList();
if( allocatedCount ) {
emptyEnoughToUse(); 
} else {
restoreBumpPtr();
}
MALLOC_ASSERT( !isNotForUse(publicFreeList), ASSERT_TEXT );
}

void Block::shareOrphaned(intptr_t binTag, unsigned index)
{
MALLOC_ASSERT( binTag, ASSERT_TEXT );
STAT_increment(getThreadId(), index, freeBlockPublic);
markOrphaned();
if ((intptr_t)nextPrivatizable==binTag) {
void* oldval;
#if FREELIST_NONBLOCKING
oldval = (void*)AtomicCompareExchange((intptr_t&)publicFreeList, (intptr_t)UNUSABLE, 0);
#else
STAT_increment(getThreadId(), ThreadCommonCounters, lockPublicFreeList);
{
MallocMutex::scoped_lock scoped_cs(publicFreeListLock);
if ( (oldval=publicFreeList)==NULL )
(uintptr_t&)(publicFreeList) = UNUSABLE;
}
#endif
if ( oldval!=NULL ) {
int count = 256;
while( (intptr_t)const_cast<Block* volatile &>(nextPrivatizable)==binTag ) {
if (--count==0) {
do_yield();
count = 256;
}
}
}
} else {
MALLOC_ASSERT( isSolidPtr(publicFreeList), ASSERT_TEXT );
}
MALLOC_ASSERT( publicFreeList!=NULL, ASSERT_TEXT );
previous = NULL;
(uintptr_t&)(nextPrivatizable) = UNUSABLE;
}

void Block::cleanBlockHeader()
{
next = NULL;
previous = NULL;
freeList = NULL;
allocatedCount = 0;
isFull = 0;
tlsPtr = NULL;

publicFreeList = NULL;
}

void Block::initEmptyBlock(TLSData *tls, size_t size)
{
unsigned int index = getIndex(size);
unsigned int objSz = getObjectSize(size);

cleanBlockHeader();
objectSize = objSz;
markOwned(tls);
bumpPtr = (FreeObject *)((uintptr_t)this + slabSize - objectSize);

nextPrivatizable = tls? (Block*)(tls->bin + index) : NULL;
TRACEF(( "[ScalableMalloc trace] Empty block %p is initialized, owner is %ld, objectSize is %d, bumpPtr is %p\n",
this, tlsPtr ? getThreadId() : -1, objectSize, bumpPtr ));
}

Block *OrphanedBlocks::get(TLSData *tls, unsigned int size)
{
unsigned int index = getIndex(size);
Block *block = bins[index].pop();
if (block) {
MALLOC_ITT_SYNC_ACQUIRED(bins+index);
block->privatizeOrphaned(tls, index);
}
return block;
}

void OrphanedBlocks::put(intptr_t binTag, Block *block)
{
unsigned int index = getIndex(block->getSize());
block->shareOrphaned(binTag, index);
MALLOC_ITT_SYNC_RELEASING(bins+index);
bins[index].push(block);
}

void OrphanedBlocks::reset()
{
for (uint32_t i=0; i<numBlockBinLimit; i++)
new (bins+i) LifoList();
}

bool OrphanedBlocks::cleanup(Backend* backend)
{
bool result = false;
for (uint32_t i=0; i<numBlockBinLimit; i++) {
Block* block = bins[i].grab();
MALLOC_ITT_SYNC_ACQUIRED(bins+i);
while (block) {
Block* next = block->next;
block->privatizePublicFreeList( true );
if (block->empty()) {
block->makeEmpty();
if (!backend->inUserPool())
removeBackRef(*(block->getBackRefIdx()));
backend->putSlabBlock(block);
result = true;
} else {
MALLOC_ITT_SYNC_RELEASING(bins+i);
bins[i].push(block);
}
block = next;
}
}
return result;
}

FreeBlockPool::ResOfGet FreeBlockPool::getBlock()
{
Block *b = (Block*)AtomicFetchStore(&head, 0);

if (b) {
size--;
Block *newHead = b->next;
lastAccessMiss = false;
FencedStore((intptr_t&)head, (intptr_t)newHead);
} else
lastAccessMiss = true;

return ResOfGet(b, lastAccessMiss);
}

void FreeBlockPool::returnBlock(Block *block)
{
MALLOC_ASSERT( size <= POOL_HIGH_MARK, ASSERT_TEXT );
Block *localHead = (Block*)AtomicFetchStore(&head, 0);

if (!localHead)
size = 0; 
else if (size == POOL_HIGH_MARK) {
Block *headToFree = localHead, *helper;
for (int i=0; i<POOL_LOW_MARK-2; i++)
headToFree = headToFree->next;
Block *last = headToFree;
headToFree = headToFree->next;
last->next = NULL;
size = POOL_LOW_MARK-1;
for (Block *currBl = headToFree; currBl; currBl = helper) {
helper = currBl->next;
if (!backend->inUserPool())
removeBackRef(currBl->backRefIdx);
backend->putSlabBlock(currBl);
}
}
size++;
block->next = localHead;
FencedStore((intptr_t&)head, (intptr_t)block);
}

bool FreeBlockPool::externalCleanup()
{
Block *helper;
bool nonEmpty = false;

for (Block *currBl=(Block*)AtomicFetchStore(&head, 0); currBl; currBl=helper) {
helper = currBl->next;
if (!backend->inUserPool())
removeBackRef(currBl->backRefIdx);
backend->putSlabBlock(currBl);
nonEmpty = true;
}
return nonEmpty;
}


void Block::makeEmpty()
{
MALLOC_ASSERT( allocatedCount==0, ASSERT_TEXT );
MALLOC_ASSERT( publicFreeList==NULL, ASSERT_TEXT );
if (!isStartupAllocObject())
STAT_increment(getThreadId(), getIndex(objectSize), freeBlockBack);

cleanBlockHeader();

nextPrivatizable = NULL;

objectSize = 0;
bumpPtr = (FreeObject *)((uintptr_t)this + slabSize);
}

inline void Bin::setActiveBlock (Block *block)
{
MALLOC_ASSERT( block->isOwnedByCurrentThread(), ASSERT_TEXT );
activeBlk = block;
}

inline Block* Bin::setPreviousBlockActive()
{
MALLOC_ASSERT( activeBlk, ASSERT_TEXT );
Block* temp = activeBlk->previous;
if( temp ) {
MALLOC_ASSERT( temp->isFull == 0, ASSERT_TEXT );
activeBlk = temp;
}
return temp;
}

inline bool Block::isOwnedByCurrentThread() const {
return tlsPtr && ownerTid.isCurrentThreadId();
}

FreeObject *Block::findObjectToFree(const void *object) const
{
FreeObject *objectToFree;
if (objectSize <= maxSegregatedObjectSize)
objectToFree = (FreeObject*)object;
else {
if ( ! isAligned(object,2*fittingAlignment) )
objectToFree = (FreeObject*)object;
else
objectToFree = findAllocatedObject(object);
MALLOC_ASSERT( isAligned(objectToFree,fittingAlignment), ASSERT_TEXT );
}
MALLOC_ASSERT( isProperlyPlaced(objectToFree), ASSERT_TEXT );

return objectToFree;
}

void TLSData::release(MemoryPool *mPool)
{
mPool->extMemPool.allLocalCaches.unregisterThread(this);
externalCleanup(&mPool->extMemPool, false);

for (unsigned index = 0; index < numBlockBins; index++) {
Block *activeBlk = bin[index].getActiveBlock();
if (!activeBlk)
continue;
Block *threadlessBlock = activeBlk->previous;
while (threadlessBlock) {
Block *threadBlock = threadlessBlock->previous;
if (threadlessBlock->empty()) {

mPool->returnEmptyBlock(threadlessBlock, false);
} else {
mPool->extMemPool.orphanedBlocks.put(intptr_t(bin+index), threadlessBlock);
}
threadlessBlock = threadBlock;
}
threadlessBlock = activeBlk;
while (threadlessBlock) {
Block *threadBlock = threadlessBlock->next;
if (threadlessBlock->empty()) {

mPool->returnEmptyBlock(threadlessBlock, false);
} else {
mPool->extMemPool.orphanedBlocks.put(intptr_t(bin+index), threadlessBlock);
}
threadlessBlock = threadBlock;
}
bin[index].resetActiveBlock();
}
}


#if MALLOC_CHECK_RECURSION



class StartupBlock : public Block {
size_t availableSize() const {
return slabSize - ((uintptr_t)bumpPtr - (uintptr_t)this);
}
static StartupBlock *getBlock();
public:
static FreeObject *allocate(size_t size);
static size_t msize(void *ptr) { return *((size_t*)ptr - 1); }
void free(void *ptr);
};

static MallocMutex startupMallocLock;
static StartupBlock *firstStartupBlock;

StartupBlock *StartupBlock::getBlock()
{
BackRefIdx backRefIdx = BackRefIdx::newBackRef(false);
if (backRefIdx.isInvalid()) return NULL;

StartupBlock *block = static_cast<StartupBlock*>(
defaultMemPool->extMemPool.backend.getSlabBlock(1));
if (!block) return NULL;

block->cleanBlockHeader();
setBackRef(backRefIdx, block);
block->backRefIdx = backRefIdx;
block->objectSize = startupAllocObjSizeMark;
block->bumpPtr = (FreeObject *)((uintptr_t)block + sizeof(StartupBlock));
return block;
}

FreeObject *StartupBlock::allocate(size_t size)
{
FreeObject *result;
StartupBlock *newBlock = NULL;
bool newBlockUnused = false;


size = alignUp(size, sizeof(size_t));
size_t reqSize = size + sizeof(size_t);

if (!firstStartupBlock || firstStartupBlock->availableSize() < reqSize) {
newBlock = StartupBlock::getBlock();
if (!newBlock) return NULL;
}
{
MallocMutex::scoped_lock scoped_cs(startupMallocLock);
if (!firstStartupBlock || firstStartupBlock->availableSize() < reqSize) {
if (!newBlock) {
newBlock = StartupBlock::getBlock();
if (!newBlock) return NULL;
}
newBlock->next = (Block*)firstStartupBlock;
if (firstStartupBlock)
firstStartupBlock->previous = (Block*)newBlock;
firstStartupBlock = newBlock;
} else
newBlockUnused = true;
result = firstStartupBlock->bumpPtr;
firstStartupBlock->allocatedCount++;
firstStartupBlock->bumpPtr =
(FreeObject *)((uintptr_t)firstStartupBlock->bumpPtr + reqSize);
}
if (newBlock && newBlockUnused)
defaultMemPool->returnEmptyBlock(newBlock, false);

*((size_t*)result) = size;
return (FreeObject*)((size_t*)result+1);
}

void StartupBlock::free(void *ptr)
{
Block* blockToRelease = NULL;
{
MallocMutex::scoped_lock scoped_cs(startupMallocLock);

MALLOC_ASSERT(firstStartupBlock, ASSERT_TEXT);
MALLOC_ASSERT(startupAllocObjSizeMark==objectSize
&& allocatedCount>0, ASSERT_TEXT);
MALLOC_ASSERT((uintptr_t)ptr>=(uintptr_t)this+sizeof(StartupBlock)
&& (uintptr_t)ptr+StartupBlock::msize(ptr)<=(uintptr_t)this+slabSize,
ASSERT_TEXT);
if (0 == --allocatedCount) {
if (this == firstStartupBlock)
firstStartupBlock = (StartupBlock*)firstStartupBlock->next;
if (previous)
previous->next = next;
if (next)
next->previous = previous;
blockToRelease = this;
} else if ((uintptr_t)ptr + StartupBlock::msize(ptr) == (uintptr_t)bumpPtr) {
FreeObject *newBump = (FreeObject*)((size_t*)ptr - 1);
MALLOC_ASSERT((uintptr_t)newBump>(uintptr_t)this+sizeof(StartupBlock),
ASSERT_TEXT);
bumpPtr = newBump;
}
}
if (blockToRelease) {
blockToRelease->previous = blockToRelease->next = NULL;
defaultMemPool->returnEmptyBlock(blockToRelease, false);
}
}

#endif 






static intptr_t mallocInitialized;   
static MallocMutex initMutex;


static char VersionString[] = "\0" TBBMALLOC_VERSION_STRINGS;

#if __TBB_WIN8UI_SUPPORT
bool GetBoolEnvironmentVariable(const char *) { return false; }
#else
bool GetBoolEnvironmentVariable(const char *name)
{
if (const char* s = getenv(name))
return strcmp(s,"0") != 0;
return false;
}
#endif

void AllocControlledMode::initReadEnv(const char *envName, intptr_t defaultVal)
{
if (!setDone) {
#if !__TBB_WIN8UI_SUPPORT
const char *envVal = getenv(envName);
if (envVal && !strcmp(envVal, "1"))
val = 1;
else
#endif
val = defaultVal;
setDone = true;
}
}

void MemoryPool::initDefaultPool()
{
long long unsigned hugePageSize = 0;
#if __linux__
if (FILE *f = fopen("/proc/meminfo", "r")) {
const int READ_BUF_SIZE = 100;
char buf[READ_BUF_SIZE];
MALLOC_STATIC_ASSERT(sizeof(hugePageSize) >= 8,
"At least 64 bits required for keeping page size/numbers.");

while (fgets(buf, READ_BUF_SIZE, f)) {
if (1 == sscanf(buf, "Hugepagesize: %llu kB", &hugePageSize)) {
hugePageSize *= 1024;
break;
}
}
fclose(f);
}
#endif
hugePages.init(hugePageSize);
}

#if USE_PTHREAD && (__TBB_SOURCE_DIRECTLY_INCLUDED || __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND)


class ShutdownSync {

intptr_t flag;
static const intptr_t skipDtor = INTPTR_MIN/2;
public:
void init() { flag = 0; }

bool threadDtorStart() {
if (flag < 0)
return false;
if (AtomicIncrement(flag) <= 0) { 
AtomicAdd(flag, -1);  
return false;
}
return true;
}
void threadDtorDone() {
AtomicAdd(flag, -1);
}
void processExit() {
if (AtomicAdd(flag, skipDtor) != 0)
SpinWaitUntilEq(flag, skipDtor);
}
};

#else

class ShutdownSync {
public:
void init() { }
bool threadDtorStart() { return true; }
void threadDtorDone() { }
void processExit() { }
};

#endif 

static ShutdownSync shutdownSync;

inline bool isMallocInitialized() {
return 2 == FencedLoad(mallocInitialized);
}

bool isMallocInitializedExt() {
return isMallocInitialized();
}


extern "C" void MallocInitializeITT() {
#if DO_ITT_NOTIFY
if (!usedBySrcIncluded)
tbb::internal::__TBB_load_ittnotify();
#endif
}


static bool initMemoryManager()
{
TRACEF(( "[ScalableMalloc trace] sizeof(Block) is %d (expected 128); sizeof(uintptr_t) is %d\n",
sizeof(Block), sizeof(uintptr_t) ));
MALLOC_ASSERT( 2*blockHeaderAlignment == sizeof(Block), ASSERT_TEXT );
MALLOC_ASSERT( sizeof(FreeObject) == sizeof(void*), ASSERT_TEXT );
MALLOC_ASSERT( isAligned(defaultMemPool, sizeof(intptr_t)),
"Memory pool must be void*-aligned for atomic to work over aligned arguments.");

#if USE_WINTHREAD
const size_t granularity = 64*1024; 
#else
const size_t granularity = sysconf(_SC_PAGESIZE);
#endif
bool initOk = defaultMemPool->
extMemPool.init(0, NULL, NULL, granularity,
false, false);
if (!initOk || !initBackRefMaster(&defaultMemPool->extMemPool.backend))
return false;
ThreadId::init();      
MemoryPool::initDefaultPool();
shutdownSync.init();
#if COLLECT_STATISTICS
initStatisticsCollection();
#endif
return true;
}


static bool doInitialization()
{
MallocMutex::scoped_lock lock( initMutex );
if (mallocInitialized!=2) {
MALLOC_ASSERT( mallocInitialized==0, ASSERT_TEXT );
mallocInitialized = 1;
RecursiveMallocCallProtector scoped;
if (!initMemoryManager()) {
mallocInitialized = 0; 
return false;
}
#ifdef  MALLOC_EXTRA_INITIALIZATION
MALLOC_EXTRA_INITIALIZATION;
#endif
#if MALLOC_CHECK_RECURSION
RecursiveMallocCallProtector::detectNaiveOverload();
#endif
MALLOC_ASSERT( mallocInitialized==1, ASSERT_TEXT );
FencedStore( mallocInitialized, 2 );
if( GetBoolEnvironmentVariable("TBB_VERSION") ) {
fputs(VersionString+1,stderr);
hugePages.printStatus();
}
}

MALLOC_ASSERT( mallocInitialized==2, ASSERT_TEXT );
return true;
}






FreeObject *Block::allocateFromFreeList()
{
FreeObject *result;

if (!freeList) return NULL;

result = freeList;
MALLOC_ASSERT( result, ASSERT_TEXT );

freeList = result->next;
MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
allocatedCount++;
STAT_increment(getThreadId(), getIndex(objectSize), allocFreeListUsed);

return result;
}

FreeObject *Block::allocateFromBumpPtr()
{
FreeObject *result = bumpPtr;
if (result) {
bumpPtr = (FreeObject *) ((uintptr_t) bumpPtr - objectSize);
if ( (uintptr_t)bumpPtr < (uintptr_t)this+sizeof(Block) ) {
bumpPtr = NULL;
}
MALLOC_ASSERT( allocatedCount < (slabSize-sizeof(Block))/objectSize, ASSERT_TEXT );
allocatedCount++;
STAT_increment(getThreadId(), getIndex(objectSize), allocBumpPtrUsed);
}
return result;
}

inline FreeObject* Block::allocate()
{
MALLOC_ASSERT( isOwnedByCurrentThread(), ASSERT_TEXT );


if ( FreeObject *result = allocateFromFreeList() ) {
return result;
}
MALLOC_ASSERT( !freeList, ASSERT_TEXT );


if ( FreeObject *result = allocateFromBumpPtr() ) {
return result;
}
MALLOC_ASSERT( !bumpPtr, ASSERT_TEXT );


isFull = 1;
return NULL;
}

size_t Block::findObjectSize(void *object) const
{
size_t blSize = getSize();
#if MALLOC_CHECK_RECURSION
if (!blSize)
return StartupBlock::msize(object);
#endif
size_t size =
blSize - ((uintptr_t)object - (uintptr_t)findObjectToFree(object));
MALLOC_ASSERT(size>0 && size<minLargeObjectSize, ASSERT_TEXT);
return size;
}

void Bin::moveBlockToFront(Block *block)
{

if (block == activeBlk) return;
outofTLSBin(block);
pushTLSBin(block);
}

void Bin::processLessUsedBlock(MemoryPool *memPool, Block *block)
{
if (block != activeBlk) {

outofTLSBin(block);
memPool->returnEmptyBlock(block, true);
} else {

block->restoreBumpPtr();
}
}

template<int LOW_MARK, int HIGH_MARK>
bool LocalLOCImpl<LOW_MARK, HIGH_MARK>::put(LargeMemoryBlock *object, ExtMemoryPool *extMemPool)
{
const size_t size = object->unalignedSize;
if (size > MAX_TOTAL_SIZE)
return false;
LargeMemoryBlock *localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0);

object->prev = NULL;
object->next = localHead;
if (localHead)
localHead->prev = object;
else {
totalSize = 0;
numOfBlocks = 0;
tail = object;
}
localHead = object;
totalSize += size;
numOfBlocks++;
if (totalSize > MAX_TOTAL_SIZE || numOfBlocks >= HIGH_MARK) {
while (totalSize > MAX_TOTAL_SIZE || numOfBlocks > LOW_MARK) {
totalSize -= tail->unalignedSize;
numOfBlocks--;
tail = tail->prev;
}
LargeMemoryBlock *headToRelease = tail->next;
tail->next = NULL;

extMemPool->freeLargeObjectList(headToRelease);
}

FencedStore((intptr_t&)head, (intptr_t)localHead);
return true;
}

template<int LOW_MARK, int HIGH_MARK>
LargeMemoryBlock *LocalLOCImpl<LOW_MARK, HIGH_MARK>::get(size_t size)
{
LargeMemoryBlock *localHead, *res=NULL;

if (size > MAX_TOTAL_SIZE)
return NULL;

if (!head || !(localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0))) {
return NULL;
}

for (LargeMemoryBlock *curr = localHead; curr; curr=curr->next) {
if (curr->unalignedSize == size) {
res = curr;
if (curr->next)
curr->next->prev = curr->prev;
else
tail = curr->prev;
if (curr != localHead)
curr->prev->next = curr->next;
else
localHead = curr->next;
totalSize -= size;
numOfBlocks--;
break;
}
}
FencedStore((intptr_t&)head, (intptr_t)localHead);
return res;
}

template<int LOW_MARK, int HIGH_MARK>
bool LocalLOCImpl<LOW_MARK, HIGH_MARK>::externalCleanup(ExtMemoryPool *extMemPool)
{
if (LargeMemoryBlock *localHead = (LargeMemoryBlock*)AtomicFetchStore(&head, 0)) {
extMemPool->freeLargeObjectList(localHead);
return true;
}
return false;
}

void *MemoryPool::getFromLLOCache(TLSData* tls, size_t size, size_t alignment)
{
LargeMemoryBlock *lmb = NULL;

size_t headersSize = sizeof(LargeMemoryBlock)+sizeof(LargeObjectHdr);
size_t allocationSize = LargeObjectCache::alignToBin(size+headersSize+alignment);
if (allocationSize < size) 
return NULL;
MALLOC_ASSERT(allocationSize >= alignment, "Overflow must be checked before.");

if (tls) {
tls->markUsed();
lmb = tls->lloc.get(allocationSize);
}
if (!lmb)
lmb = extMemPool.mallocLargeObject(this, allocationSize);

if (lmb) {
MALLOC_ASSERT(alignment >= estimatedCacheLineSize, ASSERT_TEXT);

void *alignedArea = (void*)alignUp((uintptr_t)lmb+headersSize, alignment);
uintptr_t alignedRight =
alignDown((uintptr_t)lmb+lmb->unalignedSize - size, alignment);
unsigned ptrDelta = alignedRight - (uintptr_t)alignedArea;
if (ptrDelta && tls) { 
unsigned numOfPossibleOffsets = alignment == estimatedCacheLineSize?
ptrDelta / estimatedCacheLineSize :
ptrDelta / alignment;
unsigned myCacheIdx = ++tls->currCacheIdx;
unsigned offset = myCacheIdx % numOfPossibleOffsets;

alignedArea = (void*)((uintptr_t)alignedArea + offset*alignment);
}
MALLOC_ASSERT((uintptr_t)lmb+lmb->unalignedSize >=
(uintptr_t)alignedArea+size, "Object doesn't fit the block.");
LargeObjectHdr *header = (LargeObjectHdr*)alignedArea-1;
header->memoryBlock = lmb;
header->backRefIdx = lmb->backRefIdx;
setBackRef(header->backRefIdx, header);

lmb->objectSize = size;

MALLOC_ASSERT( isLargeObject<unknownMem>(alignedArea), ASSERT_TEXT );
MALLOC_ASSERT( isAligned(alignedArea, alignment), ASSERT_TEXT );

return alignedArea;
}
return NULL;
}

void MemoryPool::putToLLOCache(TLSData *tls, void *object)
{
LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
header->backRefIdx = BackRefIdx();

if (tls) {
tls->markUsed();
if (tls->lloc.put(header->memoryBlock, &extMemPool))
return;
}
extMemPool.freeLargeObject(header->memoryBlock);
}


static void *allocateAligned(MemoryPool *memPool, size_t size, size_t alignment)
{
MALLOC_ASSERT( isPowerOfTwo(alignment), ASSERT_TEXT );

if (!isMallocInitialized())
if (!doInitialization())
return NULL;

void *result;
if (size<=maxSegregatedObjectSize && alignment<=maxSegregatedObjectSize)
result = internalPoolMalloc(memPool, alignUp(size? size: sizeof(size_t), alignment));
else if (size<minLargeObjectSize) {
if (alignment<=fittingAlignment)
result = internalPoolMalloc(memPool, size);
else if (size+alignment < minLargeObjectSize) {
void *unaligned = internalPoolMalloc(memPool, size+alignment);
if (!unaligned) return NULL;
result = alignUp(unaligned, alignment);
} else
goto LargeObjAlloc;
} else {
LargeObjAlloc:
TLSData *tls = memPool->getTLS(true);
result =
memPool->getFromLLOCache(tls, size, largeObjectAlignment>alignment?
largeObjectAlignment: alignment);
}

MALLOC_ASSERT( isAligned(result, alignment), ASSERT_TEXT );
return result;
}

static void *reallocAligned(MemoryPool *memPool, void *ptr,
size_t size, size_t alignment = 0)
{
void *result;
size_t copySize;

if (isLargeObject<ourMem>(ptr)) {
LargeMemoryBlock* lmb = ((LargeObjectHdr *)ptr - 1)->memoryBlock;
copySize = lmb->unalignedSize-((uintptr_t)ptr-(uintptr_t)lmb);
if (size <= copySize && (0==alignment || isAligned(ptr, alignment))) {
lmb->objectSize = size;
return ptr;
} else {
copySize = lmb->objectSize;
#if BACKEND_HAS_MREMAP
if (void *r = memPool->extMemPool.remap(ptr, copySize, size,
alignment<largeObjectAlignment?
largeObjectAlignment : alignment))
return r;
#endif
result = alignment ? allocateAligned(memPool, size, alignment) :
internalPoolMalloc(memPool, size);
}
} else {
Block* block = (Block *)alignDown(ptr, slabSize);
copySize = block->findObjectSize(ptr);
if (size <= copySize && (0==alignment || isAligned(ptr, alignment))) {
return ptr;
} else {
result = alignment ? allocateAligned(memPool, size, alignment) :
internalPoolMalloc(memPool, size);
}
}
if (result) {
memcpy(result, ptr, copySize<size? copySize: size);
internalPoolFree(memPool, ptr, 0);
}
return result;
}


inline bool Block::isProperlyPlaced(const void *object) const
{
return 0 == ((uintptr_t)this + slabSize - (uintptr_t)object) % objectSize;
}


FreeObject *Block::findAllocatedObject(const void *address) const
{
uint16_t offset = (uintptr_t)this + slabSize - (uintptr_t)address;
MALLOC_ASSERT( offset<=slabSize-sizeof(Block), ASSERT_TEXT );
offset %= objectSize;
return (FreeObject*)((uintptr_t)address - (offset? objectSize-offset: 0));
}


static inline BackRefIdx safer_dereference (const BackRefIdx *ptr)
{
BackRefIdx id;
#if _MSC_VER
__try {
#endif
id = *ptr;
#if _MSC_VER
} __except( GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION?
EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH ) {
id = BackRefIdx();
}
#endif
return id;
}

template<MemoryOrigin memOrigin>
bool isLargeObject(void *object)
{
if (!isAligned(object, largeObjectAlignment))
return false;
LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
BackRefIdx idx = memOrigin==unknownMem? safer_dereference(&header->backRefIdx) :
header->backRefIdx;

return idx.isLargeObject()
&& header->memoryBlock
&& (uintptr_t)header->memoryBlock < (uintptr_t)header
&& getBackRef(idx) == header;
}

static inline bool isSmallObject (void *ptr)
{
Block* expectedBlock = (Block*)alignDown(ptr, slabSize);
const BackRefIdx* idx = expectedBlock->getBackRefIdx();

bool isSmall = expectedBlock == getBackRef(safer_dereference(idx));
if (isSmall)
expectedBlock->checkFreePrecond(ptr);
return isSmall;
}


static inline bool isRecognized (void* ptr)
{
return defaultMemPool->extMemPool.backend.ptrCanBeValid(ptr) &&
(isLargeObject<unknownMem>(ptr) || isSmallObject(ptr));
}

static inline void freeSmallObject(void *object)
{

Block *block = (Block *)alignDown(object, slabSize);
block->checkFreePrecond(object);

#if MALLOC_CHECK_RECURSION
if (block->isStartupAllocObject()) {
((StartupBlock *)block)->free(object);
return;
}
#endif
if (block->isOwnedByCurrentThread()) {
block->freeOwnObject(object);
} else { 
FreeObject *objectToFree = block->findObjectToFree(object);
block->freePublicObject(objectToFree);
}
}

static void *internalPoolMalloc(MemoryPool* memPool, size_t size)
{
Bin* bin;
Block * mallocBlock;

if (!memPool) return NULL;

if (!size) size = sizeof(size_t);

TLSData *tls = memPool->getTLS(true);


if (size >= minLargeObjectSize)
return memPool->getFromLLOCache(tls, size, largeObjectAlignment);

if (!tls) return NULL;

tls->markUsed();

bin = tls->getAllocationBin(size);
if ( !bin ) return NULL;


for( mallocBlock = bin->getActiveBlock(); mallocBlock;
mallocBlock = bin->setPreviousBlockActive() ) 
{
if( FreeObject *result = mallocBlock->allocate() )
return result;
}


mallocBlock = bin->getPublicFreeListBlock();
if (mallocBlock) {
if (mallocBlock->emptyEnoughToUse()) {
bin->moveBlockToFront(mallocBlock);
}
MALLOC_ASSERT( mallocBlock->freeListNonNull(), ASSERT_TEXT );
if ( FreeObject *result = mallocBlock->allocateFromFreeList() )
return result;

TRACEF(( "[ScalableMalloc trace] Something is wrong: no objects in public free list; reentering.\n" ));
return internalPoolMalloc(memPool, size);
}


mallocBlock = memPool->extMemPool.orphanedBlocks.get(tls, size);
while (mallocBlock) {
bin->pushTLSBin(mallocBlock);
bin->setActiveBlock(mallocBlock); 
if( FreeObject *result = mallocBlock->allocate() )
return result;
mallocBlock = memPool->extMemPool.orphanedBlocks.get(tls, size);
}


mallocBlock = memPool->getEmptyBlock(size);
if (mallocBlock) {
bin->pushTLSBin(mallocBlock);
bin->setActiveBlock(mallocBlock);
if( FreeObject *result = mallocBlock->allocate() )
return result;

TRACEF(( "[ScalableMalloc trace] Something is wrong: no objects in empty block; reentering.\n" ));
return internalPoolMalloc(memPool, size);
}

TRACEF(( "[ScalableMalloc trace] No memory found, returning NULL.\n" ));
return NULL;
}

static bool internalPoolFree(MemoryPool *memPool, void *object, size_t size)
{
if (!memPool || !object) return false;

MALLOC_ASSERT(isMallocInitialized(), ASSERT_TEXT);
MALLOC_ASSERT(memPool->extMemPool.userPool() || isRecognized(object),
"Invalid pointer during object releasing is detected.");

if (size >= minLargeObjectSize || isLargeObject<ourMem>(object))
memPool->putToLLOCache(memPool->getTLS(false), object);
else
freeSmallObject(object);
return true;
}

static void *internalMalloc(size_t size)
{
if (!size) size = sizeof(size_t);

#if MALLOC_CHECK_RECURSION
if (RecursiveMallocCallProtector::sameThreadActive())
return size<minLargeObjectSize? StartupBlock::allocate(size) :
(FreeObject*)defaultMemPool->getFromLLOCache(NULL, size, slabSize);
#endif

if (!isMallocInitialized())
if (!doInitialization())
return NULL;
return internalPoolMalloc(defaultMemPool, size);
}

static void internalFree(void *object)
{
internalPoolFree(defaultMemPool, object, 0);
}

static size_t internalMsize(void* ptr)
{
if (ptr) {
MALLOC_ASSERT(isRecognized(ptr), "Invalid pointer in scalable_msize detected.");
if (isLargeObject<ourMem>(ptr)) {
LargeMemoryBlock* lmb = ((LargeObjectHdr*)ptr - 1)->memoryBlock;
return lmb->objectSize;
} else
return ((Block*)alignDown(ptr, slabSize))->findObjectSize(ptr);
}
errno = EINVAL;
return 0;
}

} 

using namespace rml::internal;

rml::MemoryPool *pool_create(intptr_t pool_id, const MemPoolPolicy *policy)
{
rml::MemoryPool *pool;
MemPoolPolicy pol(policy->pAlloc, policy->pFree, policy->granularity);

pool_create_v1(pool_id, &pol, &pool);
return pool;
}

rml::MemPoolError pool_create_v1(intptr_t pool_id, const MemPoolPolicy *policy,
rml::MemoryPool **pool)
{
if ( !policy->pAlloc || policy->version<MemPoolPolicy::TBBMALLOC_POOL_VERSION
|| !(policy->fixedPool || policy->pFree)) {
*pool = NULL;
return INVALID_POLICY;
}
if ( policy->version>MemPoolPolicy::TBBMALLOC_POOL_VERSION 
|| policy->reserved ) {
*pool = NULL;
return UNSUPPORTED_POLICY;
}
if (!isMallocInitialized())
if (!doInitialization())
return NO_MEMORY;
rml::internal::MemoryPool *memPool =
(rml::internal::MemoryPool*)internalMalloc((sizeof(rml::internal::MemoryPool)));
if (!memPool) {
*pool = NULL;
return NO_MEMORY;
}
memset(memPool, 0, sizeof(rml::internal::MemoryPool));
if (!memPool->init(pool_id, policy)) {
internalFree(memPool);
*pool = NULL;
return NO_MEMORY;
}

*pool = (rml::MemoryPool*)memPool;
return POOL_OK;
}

bool pool_destroy(rml::MemoryPool* memPool)
{
if (!memPool) return false;
bool ret = ((rml::internal::MemoryPool*)memPool)->destroy();
internalFree(memPool);

return ret;
}

bool pool_reset(rml::MemoryPool* memPool)
{
if (!memPool) return false;

return ((rml::internal::MemoryPool*)memPool)->reset();
}

void *pool_malloc(rml::MemoryPool* mPool, size_t size)
{
return internalPoolMalloc((rml::internal::MemoryPool*)mPool, size);
}

void *pool_realloc(rml::MemoryPool* mPool, void *object, size_t size)
{
if (!object)
return internalPoolMalloc((rml::internal::MemoryPool*)mPool, size);
if (!size) {
internalPoolFree((rml::internal::MemoryPool*)mPool, object, 0);
return NULL;
}
return reallocAligned((rml::internal::MemoryPool*)mPool, object, size, 0);
}

void *pool_aligned_malloc(rml::MemoryPool* mPool, size_t size, size_t alignment)
{
if (!isPowerOfTwo(alignment) || 0==size)
return NULL;

return allocateAligned((rml::internal::MemoryPool*)mPool, size, alignment);
}

void *pool_aligned_realloc(rml::MemoryPool* memPool, void *ptr, size_t size, size_t alignment)
{
if (!isPowerOfTwo(alignment))
return NULL;
rml::internal::MemoryPool *mPool = (rml::internal::MemoryPool*)memPool;
void *tmp;

if (!ptr)
tmp = allocateAligned(mPool, size, alignment);
else if (!size) {
internalPoolFree(mPool, ptr, 0);
return NULL;
} else
tmp = reallocAligned(mPool, ptr, size, alignment);

return tmp;
}

bool pool_free(rml::MemoryPool *mPool, void *object)
{
return internalPoolFree((rml::internal::MemoryPool*)mPool, object, 0);
}

rml::MemoryPool *pool_identify(void *object)
{
rml::internal::MemoryPool *pool;
if (isLargeObject<ourMem>(object)) {
LargeObjectHdr *header = (LargeObjectHdr*)object - 1;
pool = header->memoryBlock->pool;
} else {
Block *block = (Block*)alignDown(object, slabSize);
pool = block->getMemPool();
}
__TBB_ASSERT_RELEASE(pool!=defaultMemPool,
"rml::pool_identify() can't be used for scalable_malloc() etc results.");
return (rml::MemoryPool*)pool;
}

} 

using namespace rml::internal;

#if MALLOC_TRACE
static unsigned int threadGoingDownCount = 0;
#endif


void mallocThreadShutdownNotification(void* arg)
{
if (!isMallocInitialized()) return;

TRACEF(( "[ScalableMalloc trace] Thread id %d blocks return start %d\n",
getThreadId(),  threadGoingDownCount++ ));
#if USE_WINTHREAD
suppress_unused_warning(arg);
MallocMutex::scoped_lock lock(MemoryPool::memPoolListLock);
for (MemoryPool *memPool = defaultMemPool; memPool; memPool = memPool->next)
if (TLSData *tls = memPool->getTLS(false))
memPool->processThreadShutdown(tls);
#else
if (!shutdownSync.threadDtorStart()) return;
TLSData *tls = (TLSData*)arg;
tls->getMemPool()->processThreadShutdown(tls);
shutdownSync.threadDtorDone();
#endif

TRACEF(( "[ScalableMalloc trace] Thread id %d blocks return end\n", getThreadId() ));
}

#if USE_WINTHREAD
extern "C" void __TBB_mallocThreadShutdownNotification()
{
mallocThreadShutdownNotification(NULL);
}
#endif

extern "C" void __TBB_mallocProcessShutdownNotification()
{
if (!isMallocInitialized()) return;

#if  __TBB_MALLOC_LOCACHE_STAT
printf("cache hit ratio %f, size hit %f\n",
1.*cacheHits/mallocCalls, 1.*memHitKB/memAllocKB);
defaultMemPool->extMemPool.loc.reportStat(stdout);
#endif

shutdownSync.processExit();
#if __TBB_SOURCE_DIRECTLY_INCLUDED

defaultMemPool->destroy();
destroyBackRefMaster(&defaultMemPool->extMemPool.backend);
ThreadId::destroy();      
hugePages.reset();
FencedStore(mallocInitialized, 0);
#elif __TBB_USE_DLOPEN_REENTRANCY_WORKAROUND

defaultMemPool->extMemPool.hardCachesCleanup();
#endif 

#if COLLECT_STATISTICS
unsigned nThreads = ThreadId::getMaxThreadId();
for( int i=1; i<=nThreads && i<MAX_THREADS; ++i )
STAT_print(i);
#endif
if (!usedBySrcIncluded)
MALLOC_ITT_FINI_ITTLIB();
}

extern "C" void * scalable_malloc(size_t size)
{
void *ptr = internalMalloc(size);
if (!ptr) errno = ENOMEM;
return ptr;
}

extern "C" void scalable_free (void *object) {
internalFree(object);
}

#if MALLOC_ZONE_OVERLOAD_ENABLED
extern "C" void __TBB_malloc_free_definite_size(void *object, size_t size) {
internalPoolFree(defaultMemPool, object, size);
}
#endif


extern "C" void __TBB_malloc_safer_free(void *object, void (*original_free)(void*))
{
if (!object)
return;

if (FencedLoad(mallocInitialized) && defaultMemPool->extMemPool.backend.ptrCanBeValid(object)) {
if (isLargeObject<unknownMem>(object)) {
TLSData *tls = defaultMemPool->getTLS(false);

defaultMemPool->putToLLOCache(tls, object);
return;
} else if (isSmallObject(object)) {
freeSmallObject(object);
return;
}
}
if (original_free)
original_free(object);
}






extern "C" void* scalable_realloc(void* ptr, size_t size)
{
void *tmp;

if (!ptr)
tmp = internalMalloc(size);
else if (!size) {
internalFree(ptr);
return NULL;
} else
tmp = reallocAligned(defaultMemPool, ptr, size, 0);

if (!tmp) errno = ENOMEM;
return tmp;
}


extern "C" void* __TBB_malloc_safer_realloc(void* ptr, size_t sz, void* original_realloc)
{
void *tmp; 

if (!ptr) {
tmp = internalMalloc(sz);
} else if (FencedLoad(mallocInitialized) && isRecognized(ptr)) {
if (!sz) {
internalFree(ptr);
return NULL;
} else {
tmp = reallocAligned(defaultMemPool, ptr, sz, 0);
}
}
#if USE_WINTHREAD
else if (original_realloc && sz) {
orig_ptrs *original_ptrs = static_cast<orig_ptrs*>(original_realloc);
if ( original_ptrs->msize ){
size_t oldSize = original_ptrs->msize(ptr);
tmp = internalMalloc(sz);
if (tmp) {
memcpy(tmp, ptr, sz<oldSize? sz : oldSize);
if ( original_ptrs->free ){
original_ptrs->free( ptr );
}
}
} else
tmp = NULL;
}
#else
else if (original_realloc) {
typedef void* (*realloc_ptr_t)(void*,size_t);
realloc_ptr_t original_realloc_ptr;
(void *&)original_realloc_ptr = original_realloc;
tmp = original_realloc_ptr(ptr,sz);
}
#endif
else tmp = NULL;

if (!tmp) errno = ENOMEM;
return tmp;
}







extern "C" void * scalable_calloc(size_t nobj, size_t size)
{
const size_t mult_not_overflow = size_t(1) << (sizeof(size_t)*CHAR_BIT/2);
const size_t arraySize = nobj * size;

if (nobj>=mult_not_overflow || size>=mult_not_overflow) 
if (nobj && arraySize / nobj != size) {             
errno = ENOMEM;
return NULL;
}
void* result = internalMalloc(arraySize);
if (result)
memset(result, 0, arraySize);
else
errno = ENOMEM;
return result;
}





extern "C" int scalable_posix_memalign(void **memptr, size_t alignment, size_t size)
{
if ( !isPowerOfTwoAtLeast(alignment, sizeof(void*)) )
return EINVAL;
void *result = allocateAligned(defaultMemPool, size, alignment);
if (!result)
return ENOMEM;
*memptr = result;
return 0;
}

extern "C" void * scalable_aligned_malloc(size_t size, size_t alignment)
{
if (!isPowerOfTwo(alignment) || 0==size) {
errno = EINVAL;
return NULL;
}
void *tmp = allocateAligned(defaultMemPool, size, alignment);
if (!tmp) errno = ENOMEM;
return tmp;
}

extern "C" void * scalable_aligned_realloc(void *ptr, size_t size, size_t alignment)
{
if (!isPowerOfTwo(alignment)) {
errno = EINVAL;
return NULL;
}
void *tmp;

if (!ptr)
tmp = allocateAligned(defaultMemPool, size, alignment);
else if (!size) {
scalable_free(ptr);
return NULL;
} else
tmp = reallocAligned(defaultMemPool, ptr, size, alignment);

if (!tmp) errno = ENOMEM;
return tmp;
}

extern "C" void * __TBB_malloc_safer_aligned_realloc(void *ptr, size_t size, size_t alignment, void* orig_function)
{

if (!isPowerOfTwo(alignment)) {
errno = EINVAL;
return NULL;
}
void *tmp = NULL;

if (!ptr) {
tmp = allocateAligned(defaultMemPool, size, alignment);
} else if (FencedLoad(mallocInitialized) && isRecognized(ptr)) {
if (!size) {
internalFree(ptr);
return NULL;
} else {
tmp = reallocAligned(defaultMemPool, ptr, size, alignment);
}
}
#if USE_WINTHREAD
else {
orig_aligned_ptrs *original_ptrs = static_cast<orig_aligned_ptrs*>(orig_function);
if (size) {
if ( original_ptrs->aligned_msize ){
size_t oldSize = original_ptrs->aligned_msize(ptr, sizeof(void*), 0);
tmp = allocateAligned(defaultMemPool, size, alignment);
if (tmp) {
memcpy(tmp, ptr, size<oldSize? size : oldSize);
if ( original_ptrs->aligned_free ){
original_ptrs->aligned_free( ptr );
}
}
}
} else {
if ( original_ptrs->aligned_free ){
original_ptrs->aligned_free( ptr );
}
return NULL;
}
}
#else
suppress_unused_warning(orig_function);
#endif
if (!tmp) errno = ENOMEM;
return tmp;
}

extern "C" void scalable_aligned_free(void *ptr)
{
internalFree(ptr);
}






extern "C" size_t scalable_msize(void* ptr)
{
return internalMsize(ptr);
}


extern "C" size_t __TBB_malloc_safer_msize(void *object, size_t (*original_msize)(void*))
{
if (object) {
if (FencedLoad(mallocInitialized) && isRecognized(object))
return internalMsize(object);
else if (original_msize)
return original_msize(object);
}
#if USE_WINTHREAD
errno = EINVAL; 
#endif
return 0;
}


extern "C" size_t __TBB_malloc_safer_aligned_msize(void *object, size_t alignment, size_t offset, size_t (*orig_aligned_msize)(void*,size_t,size_t))
{
if (object) {
if (FencedLoad(mallocInitialized) && isRecognized(object))
return internalMsize(object);
else if (orig_aligned_msize)
return orig_aligned_msize(object,alignment,offset);
}
errno = EINVAL;
return 0;
}



extern "C" int scalable_allocation_mode(int param, intptr_t value)
{
if (param == TBBMALLOC_SET_SOFT_HEAP_LIMIT) {
defaultMemPool->extMemPool.backend.setRecommendedMaxSize((size_t)value);
return TBBMALLOC_OK;
} else if (param == USE_HUGE_PAGES) {
#if __linux__
switch (value) {
case 0:
case 1:
hugePages.setMode(value);
return TBBMALLOC_OK;
default:
return TBBMALLOC_INVALID_PARAM;
}
#else
return TBBMALLOC_NO_EFFECT;
#endif
#if __TBB_SOURCE_DIRECTLY_INCLUDED
} else if (param == TBBMALLOC_INTERNAL_SOURCE_INCLUDED) {
switch (value) {
case 0: 
case 1: 
usedBySrcIncluded = value;
return TBBMALLOC_OK;
default:
return TBBMALLOC_INVALID_PARAM;
}
#endif
}
return TBBMALLOC_INVALID_PARAM;
}

extern "C" int scalable_allocation_command(int cmd, void *param)
{
if (param)
return TBBMALLOC_INVALID_PARAM;
switch(cmd) {
case TBBMALLOC_CLEAN_THREAD_BUFFERS:
if (TLSData *tls = defaultMemPool->getTLS(false))
return tls->externalCleanup(&defaultMemPool->extMemPool,
false)?
TBBMALLOC_OK : TBBMALLOC_NO_EFFECT;
return TBBMALLOC_NO_EFFECT;
case TBBMALLOC_CLEAN_ALL_BUFFERS:
return defaultMemPool->extMemPool.hardCachesCleanup()?
TBBMALLOC_OK : TBBMALLOC_NO_EFFECT;
}
return TBBMALLOC_INVALID_PARAM;
}
