
#pragma once

#include "intrinsics.h"
#include "sysinfo.h"
#include "atomic.h"

namespace embree
{

class BarrierSys
{
public:


BarrierSys (size_t N = 0);
~BarrierSys ();

private:

BarrierSys (const BarrierSys& other) DELETED; 
BarrierSys& operator= (const BarrierSys& other) DELETED; 

public:

void init(size_t count);


void wait();

private:
void* opaque;
};


struct BarrierActive 
{
public:
BarrierActive () 
: cntr(0) {}

void reset() {
cntr.store(0);
}

void wait (size_t numThreads) 
{
cntr++;
while (cntr.load() != numThreads) 
__pause_cpu();
}

private:
std::atomic<size_t> cntr;
};


struct BarrierActiveAutoReset
{
public:
BarrierActiveAutoReset () 
: cntr0(0), cntr1(0) {}

void wait (size_t threadCount) 
{
cntr0.fetch_add(1);
while (cntr0 != threadCount) __pause_cpu();
cntr1.fetch_add(1);
while (cntr1 != threadCount) __pause_cpu();
cntr0.fetch_add(-1);
while (cntr0 != 0) __pause_cpu();
cntr1.fetch_add(-1);
while (cntr1 != 0) __pause_cpu();
}

private:
std::atomic<size_t> cntr0;
std::atomic<size_t> cntr1;
};

class LinearBarrierActive
{
public:


LinearBarrierActive (size_t threadCount = 0);
~LinearBarrierActive();

private:

LinearBarrierActive (const LinearBarrierActive& other) DELETED; 
LinearBarrierActive& operator= (const LinearBarrierActive& other) DELETED; 

public:

void init(size_t threadCount);


void wait (const size_t threadIndex);

private:
volatile unsigned char* count0;
volatile unsigned char* count1; 
volatile unsigned int mode;
volatile unsigned int flag0;
volatile unsigned int flag1;
volatile size_t threadCount;
};
}

