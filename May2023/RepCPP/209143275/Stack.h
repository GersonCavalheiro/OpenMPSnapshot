#ifndef _STACK_H
#define _STACK_H
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <omp.h>
using namespace std;

#ifdef STACK_SIZE_BIG
const int STACK_SIZE = (64*1024*1024); 
#else
const int STACK_SIZE = 1024;     
#endif


class doubleStackBase {
protected:
int stack_size;
int stack_counter;
double* stack;
public:
inline double top() {  return stack[stack_counter-1]; };
inline void pop() { stack_counter--; };
inline bool empty() { return (stack_counter==0); };
inline void init() { 
if(stack_counter>0 && stack!=NULL) delete [] stack;
stack_size=STACK_SIZE; 
stack = new double [stack_size]; 
stack_counter=0; 
assert(stack); 
#pragma omp master
cout << "info: Thread " << omp_get_thread_num() << " allocates stack with " << stack_size << " double values.\n";
};
inline int size() { return stack_counter; }
};

class longdoubleStackBase {
protected:
int stack_size;
int stack_counter;
long double* stack;
public:
inline long double top() {  return stack[stack_counter-1]; };
inline void pop() { stack_counter--; };
inline bool empty() { return (stack_counter==0); };
inline void init() { 
if(stack_counter>0 && stack!=NULL) delete [] stack;
stack_size=STACK_SIZE; 
stack = new long double [stack_size]; 
stack_counter=0; 
assert(stack); 
#pragma omp master
cout << "info: Thread " << omp_get_thread_num() << " allocates stack with " << stack_size << " long double values.\n";
};
inline int size() { return stack_counter; }
};

class doubleStack_safe : public doubleStackBase {
protected:
int in_push_mode;
public:
inline void push(double x) { assert(stack); assert(stack_counter<stack_size); stack[stack_counter++]=x; in_push_mode=1; };
inline double top() { 
assert(stack_counter);
if(in_push_mode) {
in_push_mode=0; 
#pragma omp master
cout << "info: Thread " << omp_get_thread_num() << " uses a maximum of " << stack_counter << " double values.\n";
} 
return stack[stack_counter-1]; 
};
};

class longdoubleStack : public longdoubleStackBase {
public:
inline void push(long double x) { stack[stack_counter++]=x; };
};

class doubleStack : public doubleStackBase {
public:
inline void push(double x) { stack[stack_counter++]=x; };
};

class intStackBase {
protected:
int stack_size;
int stack_counter;
int* stack;
public:
inline int top() { if(!stack_counter) return -1; return stack[stack_counter-1]; };
inline void pop() { stack_counter--; };
inline bool empty() { return (stack_counter==0); };
inline void init() { 
stack_size=STACK_SIZE; 
stack = new int [stack_size]; 
stack_counter=0; 
assert(stack); 
#pragma omp master
cout << "info: Thread " << omp_get_thread_num() << " allocates stack with " << stack_size << " int values.\n";
};
inline int size() { return stack_counter; }
};


class intStack_safe : public intStackBase {
protected:
int in_push_mode;
public:
inline void push(int a) { assert(stack); assert(stack_counter<stack_size); stack[stack_counter++]=a; in_push_mode=1; };
inline int top() { 
if(!stack_counter) return -1;
if(in_push_mode) {
in_push_mode=0; 
#pragma omp master
cout << "info: Thread " << omp_get_thread_num() << " uses a maximum of " << stack_counter << " int values.\n";
} 
return stack[stack_counter-1]; 
};
};

class intStack : public intStackBase {
public:
inline void push(int a) { stack[stack_counter++]=a; };
};

class Stackc : public intStack { };
class safeStackc : public intStack_safe { };

class Stacki : public intStack { };
class safeStacki : public intStack_safe { };

class ldStackf : public longdoubleStack { };
class Stackf : public doubleStack { };
class safeStackf : public doubleStack_safe { };

#endif
