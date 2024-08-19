






#ifndef FF_SPINLOCK_HPP
#define FF_SPINLOCK_HPP

#include <ff/sysdep.h>
#include <ff/platforms/platform.h>
#include <ff/config.hpp>


#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
#include <atomic>
namespace ff {
#define _INLINE static inline

ALIGN_TO_PRE(CACHE_LINE_SIZE) struct AtomicFlagWrapper {

#if defined(_MSC_VER)
AtomicFlagWrapper() {
F.clear();
}
#endif

bool test_and_set(std::memory_order mo) {
return F.test_and_set(mo);
}
void clear(std::memory_order mo) {
F.clear(mo);
}	
#if defined(_MSC_VER)
std::atomic_flag F;
#else
std::atomic_flag F=ATOMIC_FLAG_INIT;
#endif
}ALIGN_TO_POST(CACHE_LINE_SIZE);

typedef AtomicFlagWrapper lock_t[1];

_INLINE void init_unlocked(lock_t l) { }
_INLINE void init_locked(lock_t l)   { abort(); }
_INLINE void spin_lock(lock_t l) { 
while(l->test_and_set(std::memory_order_acquire)) ;
}
_INLINE void spin_unlock(lock_t l) { l->clear(std::memory_order_release);}
}
#else
#pragma message ("FastFlow requires a c++11 compiler")
#endif


#if !defined(__GNUC__) && defined(_MSC_VER)

#define __sync_lock_test_and_set(_PTR,_VAL)  InterlockedExchangePointer( ( _PTR ), ( _VAL ))
#endif


#if defined(__GNUC__) || defined(_MSC_VER) || defined(__APPLE__)


ALIGN_TO_PRE(CACHE_LINE_SIZE) struct CLHSpinLock {
typedef union CLHLockNode {
bool locked;
char align[CACHE_LINE_SIZE];
} CLHLockNode;

volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *Tail                    ALIGN_TO_POST(CACHE_LINE_SIZE);
volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *MyNode[MAX_NUM_THREADS] ALIGN_TO_POST(CACHE_LINE_SIZE);
volatile ALIGN_TO_PRE(CACHE_LINE_SIZE) CLHLockNode *MyPred[MAX_NUM_THREADS] ALIGN_TO_POST(CACHE_LINE_SIZE);

CLHSpinLock():Tail(NULL) {
for (int j = 0; j < MAX_NUM_THREADS; j++) {
MyNode[j] = NULL;
MyPred[j] = NULL;
}
}
~CLHSpinLock() {
if (Tail) freeAlignedMemory((void*)Tail);
for (int j = 0; j < MAX_NUM_THREADS; j++) 
if (MyNode[j]) freeAlignedMemory((void*)(MyNode[j]));
}
int init() {
if (Tail != NULL) return -1;
Tail = (CLHLockNode*)getAlignedMemory(CACHE_LINE_SIZE, sizeof(CLHLockNode));
Tail->locked = false;
for (int j = 0; j < MAX_NUM_THREADS; j++) 
MyNode[j] = (CLHLockNode*)getAlignedMemory(CACHE_LINE_SIZE, sizeof(CLHLockNode));

return 0;
}

inline void spin_lock(const int pid) {
MyNode[pid]->locked = true;
#if defined(_MSC_VER) 
MyPred[pid] = (CLHLockNode *) __sync_lock_test_and_set((void *volatile *)&Tail, (void *)MyNode[pid]);
#else 
MyPred[pid] = (CLHLockNode *) __sync_lock_test_and_set((long *)&Tail, (long)MyNode[pid]);
#endif
while (MyPred[pid]->locked == true) ;
}

inline void spin_unlock(const int pid) {
MyNode[pid]->locked = false;
MyNode[pid]= MyPred[pid];
}

} ALIGN_TO_POST(CACHE_LINE_SIZE);

typedef CLHSpinLock clh_lock_t[1];

_INLINE void init_unlocked(clh_lock_t l) { l->init();}
_INLINE void init_locked(clh_lock_t l) { abort(); }
_INLINE void spin_lock(clh_lock_t l, const int pid) { l->spin_lock(pid); }
_INLINE void spin_unlock(clh_lock_t l, const int pid) { l->spin_unlock(pid); }

#endif 

#if 0


#include <ff/mpmc/asm/abstraction_dcas.h>

#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
#include <atomic>
#endif

#ifdef __cplusplus
namespace ff {
#define _INLINE static inline
#else
#define _INLINE __forceinline
#endif

#if !defined(__GNUC__) && defined(_MSC_VER)

#define __sync_lock_test_and_set(_PTR,_VAL)  InterlockedExchangePointer( ( _PTR ), ( _VAL ))
#endif

#if defined(USE_TICKETLOCK)


#if !defined(__linux__)
#error "Ticket-lock implementation only for Linux!"
#endif

#if !defined(LOCK_PREFIX)
#define LOCK_PREFIX "lock ; "
#endif

typedef  struct {unsigned int slock;}  lock_t[1];
enum { UNLOCKED=0 };

static inline void init_unlocked(lock_t l) { l[0].slock=UNLOCKED;}
static inline void init_locked(lock_t l)   { abort(); }


static __always_inline void spin_lock(lock_t lock)
{
int inc = 0x00010000;
int tmp;

asm volatile(LOCK_PREFIX "xaddl %0, %1\n"
"movzwl %w0, %2\n\t"
"shrl $16, %0\n\t"
"1:\t"
"cmpl %0, %2\n\t"
"je 2f\n\t"
"rep ; nop\n\t"
"movzwl %1, %2\n\t"

"jmp 1b\n"
"2:"
: "+r" (inc), "+m" (lock->slock), "=&r" (tmp)
:
: "memory", "cc");
}

static __always_inline void spin_unlock(lock_t lock)
{
asm volatile(LOCK_PREFIX "incw %0"
: "+m" (lock->slock)
:
: "memory", "cc");
}


#else 





#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))

ALIGN_TO_PRE(CACHE_LINE_SIZE) struct AtomicFlagWrapper {

#ifndef _MSC_VER
AtomicFlagWrapper():F(ATOMIC_FLAG_INIT) {}
#else
AtomicFlagWrapper() {
F.clear();
}
#endif

bool test_and_set(std::memory_order mo) {
return F.test_and_set(mo);
}
void clear(std::memory_order mo) {
F.clear(mo);
}	

std::atomic_flag F;
}ALIGN_TO_POST(CACHE_LINE_SIZE);

typedef AtomicFlagWrapper lock_t[1];

_INLINE void init_unlocked(lock_t l) { }
_INLINE void init_locked(lock_t l)   { abort(); }
_INLINE void spin_lock(lock_t l) { 
while(l->test_and_set(std::memory_order_acquire)) ;
}
_INLINE void spin_unlock(lock_t l) { l->clear(std::memory_order_release);}

#else  



enum { UNLOCKED=0 };
typedef volatile int lock_t[1];

_INLINE void init_unlocked(lock_t l) { l[0]=UNLOCKED;}
_INLINE void init_locked(lock_t l)   { l[0]=!UNLOCKED;}



#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include <WinBase.h>
_INLINE void spin_lock(lock_t l) {
while (InterlockedExchange((long *)l, 1) != UNLOCKED) {	
while (l[0]) ;  
}
}

_INLINE void spin_unlock(lock_t l) {

WMB();
l[0]=UNLOCKED;
}

#else 

_INLINE void spin_lock(lock_t l) {
while (xchg((int *)l, 1) != UNLOCKED) {
while (l[0]) ;  
}
}

_INLINE void spin_unlock(lock_t l) {

WMB();
l[0]=UNLOCKED;
}
#endif 
#endif 
#endif 


#ifdef __cplusplus
} 
#endif

#endif 

#endif 
