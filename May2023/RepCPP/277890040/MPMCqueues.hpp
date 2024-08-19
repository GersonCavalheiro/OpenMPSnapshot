



#ifndef FF_MPMCQUEUE_HPP
#define FF_MPMCQUEUE_HPP




#include <cstdlib>
#include <vector>
#include <ff/buffer.hpp>
#include <ff/sysdep.h>
#include <ff/allocator.hpp>
#include <ff/platforms/platform.h>
#include <ff/mpmc/asm/abstraction_dcas.h>
#include <ff/spin-lock.hpp>




#if ( (!defined(NO_STD_C0X))  &&  !(__cplusplus >= 201103L))
#pragma message ("Define -DNO_STD_C0X to use a non c++0x/c++11 compiler")
#endif

#define NO_STD_C0X




#define CAS abstraction_cas

namespace ff {




#if !defined(NO_STD_C0X)
#include <atomic>

class MPMC_Ptr_Queue {
private:
struct element_t {
std::atomic<unsigned long> seq;
void *                     data;
};

public:

MPMC_Ptr_Queue() {}


~MPMC_Ptr_Queue() {
if (buf) {
delete [] buf;
buf=NULL;
}
}




inline bool init(size_t size) {
if (size<2) size=2;
if (!isPowerOf2(size)) size = nextPowerOf2(size);
mask = size-1;

buf = new element_t[size];
if (!buf) return false;
for(size_t i=0;i<size;++i) {
buf[i].data = NULL;
buf[i].seq.store(i,std::memory_order_relaxed);

}
pwrite.store(0,std::memory_order_relaxed);        
pread.store(0,std::memory_order_relaxed);
return true;
}


inline bool push(void *const data) {
unsigned long pw, seq;
element_t * node;
unsigned long bk = BACKOFF_MIN;
do {
pw    = pwrite.load(std::memory_order_relaxed);
node  = &buf[pw & mask];
seq   = node->seq.load(std::memory_order_acquire);


if (pw == seq) { 
if (pwrite.compare_exchange_weak(pw, pw+1, std::memory_order_relaxed))
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} else 
if (pw > seq) return false; 
} while(1);
node->data = data;
node->seq.store(seq+1,std::memory_order_release);
return true;
}


inline bool pop(void** data) {
unsigned long pr, seq;
element_t * node;
unsigned long bk = BACKOFF_MIN;

do {
pr    = pread.load(std::memory_order_relaxed);
node  = &buf[pr & mask];
seq   = node->seq.load(std::memory_order_acquire);

long diff = seq - (pr+1);
if (diff == 0) { 
if (pread.compare_exchange_weak(pr, (pr+1), std::memory_order_relaxed))
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} else { 
if (diff < 0) return false; 
}
} while(1);
*data = node->data;
node->seq.store((pr+mask+1), std::memory_order_release);
return true;
}

private:
union {
std::atomic<unsigned long>  pwrite; 
char padding1[CACHE_LINE_SIZE]; 
};
union {
std::atomic<unsigned long>  pread;  
char padding2[CACHE_LINE_SIZE]; 
};
element_t *                 buf;
unsigned long               mask;
};


#else  
#include <ff/mpmc/asm/atomic.h>

class MPMC_Ptr_Queue {
protected:

struct element_t {
atomic_long_t seq;
void *        data;
};

public:

MPMC_Ptr_Queue() {}


~MPMC_Ptr_Queue() { 
if (buf) {
freeAlignedMemory(buf);
buf = NULL;
}
}




inline bool init(size_t size) {
if (size<2) size=2;
if (!isPowerOf2(size)) size = nextPowerOf2(size);
mask = (unsigned long) (size-1);

buf=(element_t*)getAlignedMemory(longxCacheLine*sizeof(long),size*sizeof(element_t));
if (!buf) return false;
for(size_t i=0;i<size;++i) {
buf[i].data = NULL;
atomic_long_set(&buf[i].seq,long(i));
}
atomic_long_set(&pwrite,0);
atomic_long_set(&pread,0);

return true;
}


inline bool push(void *const data) {
unsigned long pw, seq;
element_t * node;
unsigned long bk = BACKOFF_MIN;

do {
pw    = atomic_long_read(&pwrite);
node  = &buf[pw & mask];
seq   = atomic_long_read(&node->seq);

if (pw == seq) {
if (abstraction_cas((volatile atom_t*)&pwrite, (atom_t)(pw+1), (atom_t)pw)==(atom_t)pw) 
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} else 
if (pw > seq) return false;

} while(1);
node->data = data;
atomic_long_set(&node->seq, (seq+1));
return true;
}


inline bool pop(void** data) {
unsigned long pr , seq;
element_t * node;
unsigned long bk = BACKOFF_MIN;

do {
pr    = atomic_long_read(&pread);
node  = &buf[pr & mask];
seq   = atomic_long_read(&node->seq);
long diff = seq - (pr+1);
if (diff == 0) {
if (abstraction_cas((volatile atom_t*)&pread, (atom_t)(pr+1), (atom_t)pr)==(atom_t)pr) 
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} else { 
if (diff < 0) return false;
}

} while(1);
*data = node->data;
atomic_long_set(&node->seq,(pr+mask+1));
return true;
}

private:
union {
atomic_long_t  pwrite;
char           padding1[CACHE_LINE_SIZE];
};
union {
atomic_long_t  pread;
char           padding2[CACHE_LINE_SIZE];
};
protected:
element_t *    buf;
unsigned long  mask;
};




class uMPMC_Ptr_Queue {
protected:
enum {DEFAULT_NUM_QUEUES=4, DEFAULT_uSPSC_SIZE=2048};

typedef void *        data_element_t;
typedef atomic_long_t sequenceP_t;
typedef atomic_long_t sequenceC_t;

public:

uMPMC_Ptr_Queue() {}


~uMPMC_Ptr_Queue() {
if (buf) {
for(size_t i=0;i<(mask+1);++i) {
if (buf[i]) delete (uSWSR_Ptr_Buffer*)(buf[i]);
}
freeAlignedMemory(buf);
buf = NULL;
}
if (seqP) freeAlignedMemory(seqP);        
if (seqC) freeAlignedMemory(seqC);
}


inline bool init(unsigned long nqueues=DEFAULT_NUM_QUEUES, size_t size=DEFAULT_uSPSC_SIZE) {
if (nqueues<2) nqueues=2;
if (!isPowerOf2(nqueues)) nqueues = nextPowerOf2(nqueues);
mask = nqueues-1;

buf=(data_element_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(data_element_t));
seqP=(sequenceP_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(sequenceP_t));
seqC=(sequenceP_t*)getAlignedMemory(longxCacheLine*sizeof(long),nqueues*sizeof(sequenceC_t));

for(size_t i=0;i<nqueues;++i) {
buf[i]= new uSWSR_Ptr_Buffer(size);
((uSWSR_Ptr_Buffer*)(buf[i]))->init();
atomic_long_set(&(seqP[i]),long(i));
atomic_long_set(&(seqC[i]),long(i));
}
atomic_long_set(&preadP,0);
atomic_long_set(&preadC,0);
return true;
}


inline bool push(void *const data) {
unsigned long pw,seq,idx;
unsigned long bk = BACKOFF_MIN;
do {
pw    = atomic_long_read(&preadP);
idx   = pw & mask;
seq   = atomic_long_read(&seqP[idx]);
if (pw == seq) {
if (abstraction_cas((volatile atom_t*)&preadP, (atom_t)(pw+1), (atom_t)pw)==(atom_t)pw) 
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} 
} while(1);
((uSWSR_Ptr_Buffer*)(buf[idx]))->push(data); 
atomic_long_set(&seqP[idx],(pw+mask+1));
return true;               
}


inline bool pop(void ** data) {
unsigned long pr,idx;
long seq;
unsigned long bk = BACKOFF_MIN;

do {
pr     = atomic_long_read(&preadC);
idx    = pr & mask;
seq    = atomic_long_read(&seqC[idx]);
if (pr == (unsigned long)seq) { 
if (atomic_long_read(&seqP[idx]) <= (unsigned long)seq) return false; 
if (abstraction_cas((volatile atom_t*)&preadC, (atom_t)(pr+1), (atom_t)pr)==(atom_t)pr) 
break;

for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
}  
} while(1);
((uSWSR_Ptr_Buffer*)(buf[idx]))->pop(data);
atomic_long_set(&seqC[idx],(pr+mask+1));
return true;
}

private:
union {
atomic_long_t  preadP;
char           padding1[CACHE_LINE_SIZE];
};
union {
atomic_long_t  preadC;
char           padding2[CACHE_LINE_SIZE];
};
protected:
data_element_t *  buf;
sequenceP_t    *  seqP;
sequenceC_t    *  seqC;
unsigned long     mask;

};





class MSqueue {
private:
enum {MSQUEUE_PTR=0 };

struct Node;

struct Pointer {
Pointer() { ptr[MSQUEUE_PTR]=0;}

inline bool operator !() {
return (ptr[MSQUEUE_PTR]==0);
}
inline Pointer& operator=(const Pointer & p) {
ptr[MSQUEUE_PTR]=p.ptr[MSQUEUE_PTR];
return *this;
}

inline Pointer& operator=(Node & node) {
ptr[MSQUEUE_PTR]=&node;
return *this;
}

inline Pointer & getNodeNext() {
return ptr[MSQUEUE_PTR]->next;
}
inline Node * getNode() { return  ptr[MSQUEUE_PTR]; }

inline bool operator==( const Pointer& r ) const {
return ((ptr[MSQUEUE_PTR]==r.ptr[MSQUEUE_PTR]));
}

inline operator volatile atom_t * () const { 
union { Node* const volatile* p1; volatile atom_t * p2;} pn;
pn.p1 = ptr;
return pn.p2; 
}
inline operator atom_t * () const { 
union { Node* const volatile* p1; atom_t * p2;} pn;
pn.p1 = ptr;
return pn.p2; 
}

inline operator atom_t () const { 
union { Node* volatile p1; atom_t p2;} pn;
pn.p1 = ptr[MSQUEUE_PTR];
return pn.p2; 
}

inline void set(Node & node) {
ptr[MSQUEUE_PTR]=&node;
}

inline void * getData() const { return ptr[MSQUEUE_PTR]->getData(); }

Node * volatile ptr[1];
} ALIGN_TO_POST(ALIGN_SINGLE_POINTER);

struct Node {
Node():data(0) { next.ptr[MSQUEUE_PTR]=0;}
Node(void * data):data(data) {
next.ptr[MSQUEUE_PTR]=0;
}

inline operator atom_t * () const { return (atom_t *)next; }

inline void   setData(void * const d) { data=d;}
inline void * getData() const { return data; }

Pointer   next;
void    * data;
} ALIGN_TO_POST(ALIGN_DOUBLE_POINTER);

Pointer  head;
long     padding1[longxCacheLine-1];
Pointer  tail;
long     padding2[longxCacheLine-1];;
FFAllocator *delayedAllocator;

private:
inline void allocnode(Pointer & p, void * data) {
union { Node * p1; void * p2;} pn;

if (delayedAllocator->posix_memalign((void**)&pn.p2,ALIGN_DOUBLE_POINTER,sizeof(Node))!=0) {
abort();
}            
new (pn.p2) Node(data);
p.set(*pn.p1);
}

inline void deallocnode( Node * n) {
n->~Node();
delayedAllocator->free(n);
}

public:
MSqueue(): delayedAllocator(NULL) { }

~MSqueue() {
if (delayedAllocator)  {
delete delayedAllocator;
delayedAllocator = NULL;
}
}

MSqueue& operator=(const MSqueue& v) { 
head=v.head;
tail=v.tail;
return *this;
}


int init() {
if (delayedAllocator) return 0;
delayedAllocator = new FFAllocator(2); 
if (!delayedAllocator) {
error("MSqueue::init, cannot allocate FFAllocator\n");
return -1;
}

Pointer dummy;
allocnode(dummy,NULL);

head = dummy;
tail = dummy;
return 1;
}

inline bool push(void * const data) {
bool done = false;

Pointer tailptr ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
Pointer next    ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
Pointer node    ALIGN_TO_POST(ALIGN_SINGLE_POINTER);
allocnode(node,data);

do {
tailptr = tail;
next    = tailptr.getNodeNext();

if (tailptr == tail) {
if (!next) { 
done = (CAS((volatile atom_t *)(tailptr.getNodeNext()), 
(atom_t)node, 
(atom_t)next) == (atom_t)next);
} else {     
CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
}
}
} while(!done);
CAS((volatile atom_t *)tail, (atom_t)node, (atom_t) tailptr);
return true;
}

inline bool  pop(void ** data) {        
bool done = false;

ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer headptr;
ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer tailptr;
ALIGN_TO_PRE(ALIGN_SINGLE_POINTER) Pointer next;

do {
headptr = head;
tailptr = tail;
next    = headptr.getNodeNext();

if (head == headptr) {
if (headptr.getNode() == tailptr.getNode()) {
if (!next) return false; 
CAS((volatile atom_t *)tail, (atom_t)next, (atom_t)tailptr);
} else {
*data = next.getData();
done = (CAS((volatile atom_t *)head, (atom_t)next, (atom_t)headptr) == (atom_t)headptr);
}
}
} while(!done);

deallocnode(headptr.getNode());
return true;
} 

inline bool empty() { 
if ((head.getNode() == tail.getNode()) && !(head.getNodeNext()))
return true;
return false;            
}
};






class multiSWSR {
protected:
enum {DEFAULT_NUM_QUEUES=4, DEFAULT_uSPSC_SIZE=2048};

public:
multiSWSR() {}

~multiSWSR() {
if (buf) {
for(size_t i=0;i<(mask+1);++i) {
if (buf[i]) delete buf[i];
}
freeAlignedMemory(buf);
buf = NULL;
}
if (PLock) freeAlignedMemory(PLock);        
if (CLock) freeAlignedMemory(CLock);
}

inline bool init(unsigned long nqueues=DEFAULT_NUM_QUEUES, size_t size=DEFAULT_uSPSC_SIZE) {
if (nqueues<2) nqueues=2;
if (!isPowerOf2(nqueues)) nqueues = nextPowerOf2(nqueues);
mask = nqueues-1;

buf=(uSWSR_Ptr_Buffer**)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(uSWSR_Ptr_Buffer*));
PLock=(CLHSpinLock*)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(CLHSpinLock));
CLock=(CLHSpinLock*)getAlignedMemory(CACHE_LINE_SIZE,nqueues*sizeof(CLHSpinLock));

for(size_t i=0;i<nqueues;++i) {
buf[i]= new uSWSR_Ptr_Buffer(size);
buf[i]->init();
PLock[i].init();
CLock[i].init();
}
atomic_long_set(&count, 0);
atomic_long_set(&enqueue,0);
atomic_long_set(&dequeue,0);
return true;
}

inline bool push(void *const data, int tid) {
long q = atomic_long_inc_return(&enqueue) & mask;
PLock[q].spin_lock(tid);
buf[q]->push(data);
PLock[q].spin_unlock(tid);
atomic_long_inc(&count);
return true;
}

inline bool pop(void ** data, int tid) {
if (!atomic_long_read(&count))  return false; 

long q = atomic_long_inc_return(&dequeue) & mask;
CLock[q].spin_lock(tid);
bool r = buf[q]->pop(data);
CLock[q].spin_unlock(tid);
if (r) { atomic_long_dec(&count); return true;}
return false;
}

private:
union {
atomic_long_t  enqueue;
char           padding1[CACHE_LINE_SIZE];
};
union {
atomic_long_t  dequeue;
char           padding2[CACHE_LINE_SIZE];
};
union {
atomic_long_t  count;
char           padding3[CACHE_LINE_SIZE];
};
protected:
uSWSR_Ptr_Buffer **buf;
CLHSpinLock *PLock;    
CLHSpinLock *CLock;    
size_t   mask;
};



template <typename Q>
class scalableMPMCqueue {
public:
enum {DEFAULT_POOL_SIZE=4};

scalableMPMCqueue() {
atomic_long_set(&enqueue,0);
atomic_long_set(&count,0);

#if !defined(MULTI_MPMC_RELAX_FIFO_ORDERING)
atomic_long_set(&dequeue,1);
#else
atomic_long_set(&dequeue,0);
#endif
}

int init(size_t poolsize = DEFAULT_POOL_SIZE) {
if (poolsize > pool.size()) {
pool.resize(poolsize);
}


return 1;
}

inline bool push(void * const data) {
long q = atomic_long_inc_return(&enqueue) % pool.size();
bool r = pool[q].push(data);
if (r) atomic_long_inc(&count);
return r;
}

inline bool  pop(void ** data) {      
if (!atomic_long_read(&count))  return false; 
#if !defined(MULTI_MPMC_RELAX_FIFO_ORDERING)
unsigned long bk = BACKOFF_MIN;
long q, q1;
do {
q  = atomic_long_read(&dequeue), q1 = atomic_long_read(&enqueue);
if (q > q1) return false;
if (CAS((volatile atom_t *)&dequeue, (atom_t)(q+1), (atom_t)q) == (atom_t)q) break;
for(volatile unsigned i=0;i<bk;++i) ;
bk <<= 1;
bk &= BACKOFF_MAX;
} while(1);

q %= pool.size(); 
if (pool[q].pop(data)) {
atomic_long_dec(&count);
return true;
}
return false;

#else  
long q = atomic_long_inc_return(&dequeue) % pool.size();
bool r = pool[q].pop(data);
if (r) { atomic_long_dec(&count); return true;}
return false;
#endif        
}

inline bool empty() {
for(size_t i=0;i<pool.size();++i)
if (!pool[i].empty()) return false;
return true;
}
private:
atomic_long_t enqueue;
long padding1[longxCacheLine-sizeof(atomic_long_t)];
atomic_long_t dequeue;
long padding2[longxCacheLine-sizeof(atomic_long_t)];
atomic_long_t count;
long padding3[longxCacheLine-sizeof(atomic_long_t)];
protected:
std::vector<Q> pool;
};


class multiMSqueue: public scalableMPMCqueue<MSqueue> {
public:

multiMSqueue(size_t poolsize = scalableMPMCqueue<MSqueue>::DEFAULT_POOL_SIZE) {
if (! scalableMPMCqueue<MSqueue>::init(poolsize)) {
error("multiMSqueue init ERROR\n");
abort();
}

for(size_t i=0;i<poolsize;++i)
if (pool[i].init()<0) {
error("multiMSqueue init ERROR\n");
abort();
}
}
};




#endif 




#if 0

typedef struct{
unsigned long data;
unsigned long next;        
long padding1[64-2*sizeof(unsigned long)];
}utMPMC_list_node_t;

typedef struct{

utMPMC_list_node_t* head;
long padding0[64-sizeof(unsigned long)];

utMPMC_list_node_t* tail;        
long padding1[64-sizeof(unsigned long)];
}utMPMC_list_info_t;

typedef struct{

utMPMC_list_info_t l; 

unsigned long s;       
long padding0[64-sizeof(unsigned long)]; 
}utMPMC_VB_note_t;

#if !defined(NEXT_SMALLEST_2_POW)
#define NEXT_SMALLEST_2_POW(A) (1 << (32 - __builtin_clz((A)-1)))
#endif

#if !defined(VOLATILE_READ)
#define VOLATILE_READ(X)  (*(volatile typeof(X)*)&X)

#if !defined(OPTIMIZED_MOD_ON_2_POW)
#define OPTIMIZED_MOD_ON_2_POW(X,Y) ((X) & (Y))
#endif

#define IS_WRITABLE(STATUS,MYEQC) (STATUS==MYEQC)
#define WRITABLE_STATUS(STATUS,MYEQC) (MYEQC)
#define UPDATE_AFTER_WRITE(STATUS) (STATUS+1)

#define IS_READABLE(STATUS,MYDQC) (STATUS==MYDQC+1)
#define READABLE_STATUS(STATUS,MYDQC) (MYDQC+1)
#define UPDATE_AFTER_READ(STATUS,LEN) (STATUS+LEN-1)
#endif

template <typename Q>
class utMPMC_VB {
public:
enum {DEFAULT_POOL_SIZE=4};

utMPMC_VB() {
dqc =0;
eqc = 0;

dqc = 0;
eqc = 0;
}

int init(size_t vector_len) {

len_v = NEXT_SMALLEST_2_POW(vector_len);
len_v_minus_one = len_v-1;

int done = posix_memalign((void **) v, longxCacheLine,
sizeof(utMPMC_VB_note_t) * len_v);
if (done != 0) {
return 0;
}
int i = 0;
for (i = 0; i < len_v; i++) {
v[i].s = i;
utMPMC_list_node_t * new_node;
do{new_node = (utMPMC_list_node_t *)
malloc (sizeof(utMPMC_list_node_t));}while(new_node);
new_node->data=NULL;
new_node->next=NULL;
v[i].l.tail=new_node;
v[i].l.head=new_node;
}

return 1;
}

inline bool push(void * const p) {
utMPMC_list_node_t * new_node;
do{new_node = (utMPMC_list_node_t *) 
malloc (sizeof(utMPMC_list_node_t));}while(new_node);
new_node->data= (unsigned long) p;
new_node->next=NULL;

unsigned long myEQC = __sync_fetch_and_add (&eqc, 1UL);;
unsigned long myI = OPTIMIZED_MOD_ON_2_POW(myEQC, len_v_minus_one);

unsigned long target_status = WRITABLE_STATUS(target_status, myEQC);
do{}while(VOLATILE_READ(v[myI].s) != target_status);


v[myI].l.tail->next = new_node;
v[myI].l.tail = new_node;
target_status = UPDATE_AFTER_WRITE(target_status);

__sync_synchronize();
v[myI].s = target_status;

return true;
}

inline bool  pop(void ** ret_val) {      
for (;;) {
unsigned long myDQC = VOLATILE_READ(dqc);
unsigned long myI = OPTIMIZED_MOD_ON_2_POW(myDQC, len_v_minus_one);
unsigned long target_status = v[myI].s;


if (IS_READABLE(target_status,myDQC) && (v[myI].l.tail!=v[myI].l.head)) {
int atomic_result = __sync_bool_compare_and_swap(&dqc, myDQC,
myDQC + 1);
if (atomic_result) {

utMPMC_list_node_t* to_be_remoed =  v[myI].l.head;

v[myI].l.head = v[myI].l.head->next;

*ret_val = v[myI].l.head->data;

target_status = UPDATE_AFTER_READ(target_status,len_v);
__sync_synchronize();                
v[myI].s = target_status;
free(to_be_remoed);
return true;
} else {
continue;
}
} else {

if (myDQC != VOLATILE_READ(dqc)) {
continue;
}
if (VOLATILE_READ(eqc) != VOLATILE_READ(dqc)) {
continue;
}

return false;
}
}

return true;
}

private:
long padding0[64 - sizeof(unsigned long)];
unsigned long eqc;
long padding1[64 - sizeof(unsigned long)];
unsigned long dqc;
long padding2[64 - sizeof(unsigned long)];
unsigned long len_v;
unsigned long len_v_minus_one;
utMPMC_VB_note_t * v;    
long padding3[64 - 3*sizeof(unsigned long)];
};



#endif

} 

#endif 
