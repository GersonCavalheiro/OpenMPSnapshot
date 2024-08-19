



#pragma once

#include <cstdlib>

namespace SpMP {

class MemoryPool {
public :
MemoryPool();

explicit MemoryPool(size_t sz);

~MemoryPool();

void initialize(size_t sz);

void finalize();

void setHead(size_t offset);

void setTail(size_t offset);

size_t getHead() const;

size_t getTail() const;

void *allocate(size_t sz, int align = 64);

void *allocateFront(size_t sz, int align = 64);

void deallocateAll();

template<typename T>
T *allocate(size_t cnt) {
return (T *) allocate(sizeof(T) * cnt);
}

template<typename T>
T *allocateFront(size_t cnt) {
return (T *) allocateFront(sizeof(T) * cnt);
}


bool contains(const void *ptr) const;

static MemoryPool *getSingleton();

#ifdef HBW_ALLOC
static MemoryPool *getMemkindSingleton();
#endif

private :
size_t size_;
size_t head_, tail_;
char *buffer_;
};

} 
