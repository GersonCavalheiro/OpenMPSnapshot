#pragma once

#include <assert.h>
#include <mutex>
#include <iostream>

#define MAXTHREADS	64 
#define BADRANGE	0

struct Data;
class Ranges;

Data classify(Data &D, const Ranges &R, unsigned int numt);

class alignas(32) Counter { 
public:
Counter(unsigned int num=MAXTHREADS) {
_numcount = num;
_counts = new unsigned int[num];
assert(_counts != NULL);
zero();
}

void zero() { 
for(int i=0; i<_numcount; i++)
_counts[i] = 0;
}

void increase(unsigned int id) { 
assert(id < _numcount);
_counts[id]++;
}

void xincrease(unsigned int id) { 
assert(id < _numcount);
const std::lock_guard<std::mutex> lock(cmutex);
_counts[id]++;
}

unsigned int get(unsigned int id) const { 
assert(id < _numcount);
return _counts[id];
}

void inspect() {
std::cout << "Subcounts -- ";
for(int i=0; i<_numcount; i++)
std::cout << i << ":" << _counts[i] << " ";
std::cout << "\n";
}

private:
unsigned volatile int *_counts;
unsigned int _numcount; 
std::mutex cmutex;
};

struct Range { 

Range(int a=1, int b=0) { 
lo = a;
hi = b;
}

bool within(int val) const { 
return(lo <= val && val <= hi);
}

bool strictlyin(int val) const { 
return(lo < val && val < hi);
}

int lo;
int hi; 
};

class Ranges {
public:
Ranges() { 
_num = 1;
_ranges = new Range(1, 0); 
}

Ranges& operator+=(const Range range){ 
if(newrange(range)) { 
Range *oranges = _ranges;
_ranges = new Range[_num+1];
assert(NULL != _ranges);
for(int r=0; r<_num; r++) { 
set(r, oranges[r].lo, oranges[r].hi); 
}
set(_num++, range.lo, range.hi); 
}
return *this;
}

int range(int val, bool strict = false) const { 
if(strict) {
for(int r=0; r<_num; r++) 
if(_ranges[r].strictlyin(val))
return r;
} else {
for(int r=0; r<_num; r++) 
if(_ranges[r].within(val))
return r;
}
return BADRANGE; 
}

void inspect() {
for(int r=0; r<_num; r++) { 
std::cout << r << "," << &_ranges[r] << ": " << _ranges[r].lo << ", " << _ranges[r].hi << "\n"; 
}

}

int num() const { return _num; }

private:
Range *_ranges;
int   _num;

void set(int i, int lo, int hi) { 
if(i < _num) {
_ranges[i].lo = lo;
_ranges[i].hi = hi;
}
}

bool newrange(const Range r) { 
return (range(r.lo, true) == BADRANGE && range(r.hi, true) == BADRANGE); 
}
};

struct Data {

struct Item {
int key;
int value = -1;
};

unsigned int ndata = 0;
Item *data = NULL;

Data(int n) { 
ndata = n;
data = new Item[n];
assert(NULL != data);
}

void reset() {
for(int i=0; i<ndata; i++)
data[i].value = -1;
}
void inspect() {
for(int i=0; i<ndata; i++)
std::cout << i << ": " << data[i].key << " -- " << data[i].value <<"\n";
}
};

