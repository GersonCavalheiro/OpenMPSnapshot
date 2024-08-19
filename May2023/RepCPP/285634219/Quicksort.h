

#ifndef QUICKSORT_H
#define QUICKSORT_H

template <class T>
T median_host(T x1, T x2, T x3) {
if (x1 < x2) {
if (x2 < x3) {
return x2;
} else {
if (x1 < x3) {
return x3;
} else {
return x1;
}
}
} else { 
if (x1 < x3) {
return x1;
} else { 
if (x2 < x3) {
return x2;
} else {
return x3;
}
}
}
}

#pragma omp declare target
template <typename T, typename P>
T Select (T a, T b, P c) {
return c ? b : a;
}

uint median(uint x1, uint x2, uint x3) {
if (x1 < x2) {
if (x2 < x3) {
return x2;
} else {
return Select(x1, x3, x1 < x3);
}
} else { 
if (x1 < x3) {
return x1;
} else { 
return Select(x2, x3, x2 < x3);
}
}
}
#pragma omp end declare target

#define TRUST_BUT_VERIFY 1
#ifdef CPU_DEVICE
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#else
#ifdef NVIDIA_GPU
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128
#define LQSORT_LOCAL_WORKGROUP_SIZE   256 
#define SORT_THRESHOLD                512 
#else 
#define QUICKSORT_BLOCK_SIZE         1728 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#endif 
#endif

#define EMPTY_RECORD             42

template <class T>
struct work_record {
uint start;
uint end;
T    pivot;
uint direction;

work_record() : 
start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD) {}
work_record(uint s, uint e, T p, uint d) : 
start(s), end(e), pivot(p), direction(d) {}
};


typedef struct parent_record {
uint sstart, send, oldstart, oldend, blockcount; 
parent_record(uint ss, uint se, uint os, uint oe, uint bc) : 
sstart(ss), send(se), oldstart(os), oldend(oe), blockcount(bc) {}
} parent_record;

template <class T>
struct block_record {
uint start;
uint end;
T    pivot;
uint direction;
uint parent;
block_record() : start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD), parent(0) {}
block_record(uint s, uint e, T p, uint d, uint prnt) : 
start(s), end(e), pivot(p), direction(d), parent(prnt) {}
};
#endif 
