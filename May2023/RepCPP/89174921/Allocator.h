#pragma once
#include <cstddef>

struct BlockHeader
{
bool state; 
size_t blockSize;
size_t prevBlockSize;
};

class Allocator
{

public:
Allocator(const int n);
void *mem_alloc(size_t size); 
void *mem_realloc(void *addr, size_t size); 
void mem_free(void *addr); 
void mem_dump(); 
size_t getBlockHeaderSize();

private:
size_t bHSize; 
BlockHeader *begin; 
int *endOfMemory; 

BlockHeader *nextBlockHeader(BlockHeader *current); 
BlockHeader *previousBlockHeader(BlockHeader *current); 
bool isLast(BlockHeader *h); 

void mergeWithNext(BlockHeader *current, BlockHeader *next);
void mergeWithPrevious(BlockHeader *previous, BlockHeader *current);
void mergeWithPrevious(BlockHeader *previous, BlockHeader *current, BlockHeader *next);
void mergeBoth(BlockHeader *previous, BlockHeader *current, BlockHeader *next);


void copyData(void *from, void *to, size_t quantity); 
void *searchNewBlock(void *addr, size_t size); 

void *expandLeft(void *addr, size_t size);
void *expandRight(void *addr, size_t size);
void *expandBoth(void *addr, size_t size);

void initBlockHeader(BlockHeader *bh, bool state, size_t previous, size_t size, int mask); 
void *separateOnUseAndFree(BlockHeader *current, size_t size);
void *getBlock(BlockHeader *h);
};
