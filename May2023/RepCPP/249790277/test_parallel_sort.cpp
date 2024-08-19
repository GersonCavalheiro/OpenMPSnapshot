

#include "harness_defs.h"
#include "tbb/parallel_sort.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_vector.h"
#include "harness.h"
#include <math.h>
#include <vector>
#include <exception>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4530)
#endif

#include <algorithm>
#include <iterator>
#include <functional>
#include <string>
#include <cstring>

#if !TBB_USE_EXCEPTIONS && _MSC_VER
#pragma warning (pop)
#endif


class Minimal {
int val;
public:
Minimal() {}
void set_val(int i) { val = i; }
static bool CompareWith (const Minimal &a, const Minimal &b) {
return (a.val < b.val);
}
static bool AreEqual( Minimal &a,  Minimal &b) {
return a.val == b.val;
}
};

class MinimalCompare {
public:
bool operator() (const Minimal &a, const Minimal &b) const {
return Minimal::CompareWith(a,b);
}
};

template<typename RandomAccessIterator>
bool Validate(RandomAccessIterator a, RandomAccessIterator b, size_t n) {
for (size_t i = 0; i < n; i++) {
ASSERT( a[i] == b[i], NULL );
}
return true;
}

template<>
bool Validate<std::string *>(std::string * a, std::string * b, size_t n) {
for (size_t i = 0; i < n; i++) {
if ( Verbose && a[i] != b[i]) {
for (size_t j = 0; j < n; j++) {
REPORT("a[%llu] == %s and b[%llu] == %s\n", static_cast<unsigned long long>(j), a[j].c_str(), static_cast<unsigned long long>(j), b[j].c_str());
}
}
ASSERT( a[i] == b[i], NULL );
}
return true;
}

template<>
bool Validate<Minimal *>(Minimal *a, Minimal *b, size_t n) {
for (size_t i = 0; i < n; i++) {
ASSERT( Minimal::AreEqual(a[i],b[i]), NULL );
}
return true;
}

template<>
bool Validate<tbb::concurrent_vector<Minimal>::iterator>(tbb::concurrent_vector<Minimal>::iterator a,
tbb::concurrent_vector<Minimal>::iterator b, size_t n) {
for (size_t i = 0; i < n; i++) {
ASSERT( Minimal::AreEqual(a[i],b[i]), NULL );
}
return true;
}

static std::string test_type;



template < typename RandomAccessIterator, typename Compare >
bool init_iter(RandomAccessIterator iter, RandomAccessIterator sorted_list, size_t n, const Compare &compare, bool reset) {
static char test_case = 0;
const char num_cases = 3;

if (reset) test_case = 0;

if (test_case < num_cases) {
switch(test_case) {
case 0:

test_type = "sin";
for (size_t i = 0; i < n; i++)
iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(sin(float(i)));
break;
case 1:

test_type = "pre-sorted";
for (size_t i = 0; i < n; i++)
iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(i);
break;
case 2:

test_type = "reverse-sorted";
for (size_t i = 0; i < n; i++)
iter[i] = sorted_list[i] = static_cast<typename std::iterator_traits< RandomAccessIterator >::value_type>(n - i);
break;
}

std::sort(sorted_list, sorted_list + n, compare);
test_case++;
return true;
}
return false;
}

template < typename T, typename Compare >
bool init_iter(T * iter, T * sorted_list, size_t n, const Compare &compare, bool reset) {
static char test_case = 0;
const char num_cases = 3;

if (reset) test_case = 0;

if (test_case < num_cases) {
switch(test_case) {
case 0:

test_type = "sin";
for (size_t i = 0; i < n; i++) {
iter[i] = T(sin(float(i)));
sorted_list[i] = T(sin(float(i)));
}
break;
case 1:

test_type = "pre-sorted";
for (size_t i = 0; i < n; i++) {
iter[i] = T(i);
sorted_list[i] = T(i);
}
break;
case 2:

test_type = "reverse-sorted";
for (size_t i = 0; i < n; i++) {
iter[i] = T(n - i);
sorted_list[i] = T(n - i);
}
break;
}

std::sort(sorted_list, sorted_list + n, compare);
test_case++;
return true;
}
return false;
}




template < >
bool init_iter(Minimal* iter, Minimal * sorted_list, size_t n, const MinimalCompare &compare, bool reset) {
static char test_case = 0;
const char num_cases = 3;

if (reset) test_case = 0;

if (test_case < num_cases) {
switch(test_case) {
case 0:

test_type = "sin";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int( sin( float(i) ) * 1000.f) );
sorted_list[i].set_val( int ( sin( float(i) ) * 1000.f) );
}
break;
case 1:

test_type = "pre-sorted";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int(i) );
sorted_list[i].set_val( int(i) );
}
break;
case 2:

test_type = "reverse-sorted";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int(n-i) );
sorted_list[i].set_val( int(n-i) );
}
break;
}
std::sort(sorted_list, sorted_list + n, compare);
test_case++;
return true;
}
return false;
}



template < >
bool init_iter(tbb::concurrent_vector<Minimal>::iterator iter, tbb::concurrent_vector<Minimal>::iterator sorted_list,
size_t n, const MinimalCompare &compare, bool reset) {
static char test_case = 0;
const char num_cases = 3;

if (reset) test_case = 0;

if (test_case < num_cases) {
switch(test_case) {
case 0:

test_type = "sin";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int( sin( float(i) ) * 1000.f) );
sorted_list[i].set_val( int ( sin( float(i) ) * 1000.f) );
}
break;
case 1:

test_type = "pre-sorted";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int(i) );
sorted_list[i].set_val( int(i) );
}
break;
case 2:

test_type = "reverse-sorted";
for (size_t i = 0; i < n; i++) {
iter[i].set_val( int(n-i) );
sorted_list[i].set_val( int(n-i) );
}
break;
}
std::sort(sorted_list, sorted_list + n, compare);
test_case++;
return true;
}
return false;
}



template<>
bool init_iter(std::string *iter, std::string *sorted_list, size_t n, const std::less<std::string> &compare, bool reset) {
static char test_case = 0;
const char num_cases = 1;

if (reset) test_case = 0;

if (test_case < num_cases) {
switch(test_case) {
case 0:

test_type = "sin";
for (size_t i = 0; i < n; i++) {
char buffer[20];
#if _MSC_VER && __STDC_SECURE_LIB__>=200411
sprintf_s(buffer, sizeof(buffer), "%f", float(sin(float(i))));
#else
sprintf(buffer, "%f", float(sin(float(i))));
#endif
sorted_list[i] = iter[i] = std::string(buffer);
}
break;
}
std::sort(sorted_list, sorted_list + n, compare);
test_case++;
return true;
}
return false;
}

static size_t current_p;

static std::string current_type;


template<typename RandomAccessIterator, typename Compare>
bool parallel_sortTest(size_t n, RandomAccessIterator iter, RandomAccessIterator sorted_list, const Compare *comp) {
bool passed = true;

Compare local_comp;

init_iter(iter, sorted_list, n, local_comp, true);
do {
REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));
if (comp != NULL) {
tbb::parallel_sort(iter, iter + n, local_comp );
} else {
tbb::parallel_sort(iter, iter + n );
}
if (!Validate(iter, sorted_list, n))
passed = false;
REMARK("passed\n");
} while (init_iter(iter, sorted_list, n, local_comp, false));
return passed;
}

template<>
bool parallel_sortTest(size_t n, Minimal * iter, Minimal * sorted_list, const MinimalCompare *compare) {
bool passed = true;

if (compare == NULL) return passed;

init_iter(iter, sorted_list, n, *compare, true);
do {
REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));

tbb::parallel_sort(iter, iter + n, *compare );

if (!Validate(iter, sorted_list, n))
passed = false;
REMARK("passed\n");
} while (init_iter(iter, sorted_list, n, *compare, false));
return passed;
}

template<>
bool parallel_sortTest(size_t n, tbb::concurrent_vector<Minimal>::iterator iter,
tbb::concurrent_vector<Minimal>::iterator sorted_list, const MinimalCompare *compare) {
bool passed = true;

if (compare == NULL) return passed;

init_iter(iter, sorted_list, n, *compare, true);
do {
REMARK("%s %s p=%llu n=%llu :",current_type.c_str(), test_type.c_str(),
static_cast<unsigned long long>(current_p), static_cast<unsigned long long>(n));

tbb::parallel_sort(iter, iter + n, *compare );

if (!Validate(iter, sorted_list, n))
passed = false;
REMARK("passed\n");
} while (init_iter(iter, sorted_list, n, *compare, false));
return passed;
}


void Flog() {

const size_t N = 50000;

Minimal *minimal_array = new Minimal[N];
Minimal *minimal_array_2 = new Minimal[N];
MinimalCompare minimal_less;

float *float_array = new float[N];
float *float_array_2 = new float[N];
std::less<float> float_less;

tbb::concurrent_vector<float> float_cv1;
tbb::concurrent_vector<float> float_cv2;
float_cv1.grow_to_at_least(N);
float_cv2.grow_to_at_least(N);

std::string *string_array = new std::string[N];
std::string *string_array_2 = new std::string[N];
std::less<std::string> string_less;

tbb::concurrent_vector<Minimal> minimal_cv1;
tbb::concurrent_vector<Minimal> minimal_cv2;
minimal_cv1.grow_to_at_least(N);
minimal_cv2.grow_to_at_least(N);



current_type = "Minimal(less)";
parallel_sortTest(0, minimal_array, minimal_array_2, &minimal_less);
parallel_sortTest(1, minimal_array, minimal_array_2, &minimal_less);
parallel_sortTest(10, minimal_array, minimal_array_2, &minimal_less);
parallel_sortTest(9999, minimal_array, minimal_array_2, &minimal_less);
parallel_sortTest(50000, minimal_array, minimal_array_2, &minimal_less);

current_type = "float (no less)";
parallel_sortTest(0, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
parallel_sortTest(1, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
parallel_sortTest(10, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
parallel_sortTest(9999, float_array, float_array_2, static_cast<std::less<float> *>(NULL));
parallel_sortTest(50000, float_array, float_array_2, static_cast<std::less<float> *>(NULL));

current_type = "float (less)";
parallel_sortTest(0, float_array, float_array_2, &float_less);
parallel_sortTest(1, float_array, float_array_2, &float_less);
parallel_sortTest(10, float_array, float_array_2, &float_less);
parallel_sortTest(9999, float_array, float_array_2, &float_less);
parallel_sortTest(50000, float_array, float_array_2, &float_less);

current_type = "concurrent_vector<float> (no less)";
parallel_sortTest(0, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
parallel_sortTest(1, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
parallel_sortTest(10, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
parallel_sortTest(9999, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));
parallel_sortTest(50000, float_cv1.begin(), float_cv2.begin(), static_cast<std::less<float> *>(NULL));

current_type = "concurrent_vector<float> (less)";
parallel_sortTest(0, float_cv1.begin(), float_cv2.begin(), &float_less);
parallel_sortTest(1, float_cv1.begin(), float_cv2.begin(), &float_less);
parallel_sortTest(10, float_cv1.begin(), float_cv2.begin(), &float_less);
parallel_sortTest(9999, float_cv1.begin(), float_cv2.begin(), &float_less);
parallel_sortTest(50000, float_cv1.begin(), float_cv2.begin(), &float_less);

current_type = "string (no less)";
parallel_sortTest(0, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
parallel_sortTest(1, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
parallel_sortTest(10, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
parallel_sortTest(9999, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));
parallel_sortTest(50000, string_array, string_array_2, static_cast<std::less<std::string> *>(NULL));

current_type = "string (less)";
parallel_sortTest(0, string_array, string_array_2, &string_less);
parallel_sortTest(1, string_array, string_array_2, &string_less);
parallel_sortTest(10, string_array, string_array_2, &string_less);
parallel_sortTest(9999, string_array, string_array_2, &string_less);
parallel_sortTest(50000, string_array, string_array_2, &string_less);

current_type = "concurrent_vector<Minimal> (less)";
parallel_sortTest(0, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
parallel_sortTest(1, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
parallel_sortTest(10, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
parallel_sortTest(9999, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);
parallel_sortTest(50000, minimal_cv1.begin(), minimal_cv2.begin(), &minimal_less);

delete [] minimal_array;
delete [] minimal_array_2;

delete [] float_array;
delete [] float_array_2;

delete [] string_array;
delete [] string_array_2;
}

const int elements = 10000;

void rand_vec(std::vector<int> &v) {
for (int i=0; i<elements; ++i) {
(v.push_back(rand()%elements*10));
}
}

void range_sort_test() {
std::vector<int> v;

typedef std::vector<int>::iterator itor;
rand_vec(v);
tbb::parallel_sort(v.begin(), v.end());
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
v.clear();

rand_vec(v);
tbb::parallel_sort(v.begin(), v.end(), std::greater<int>());
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
v.clear();

rand_vec(v);
tbb::parallel_sort(v);
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
v.clear();

rand_vec(v);
tbb::parallel_sort(v, std::greater<int>());
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
v.clear();

rand_vec(v);
tbb::parallel_sort(tbb::blocked_range<std::vector<int>::iterator>(v.begin(), v.end()));
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a <= *(a+1), "v not sorted");
v.clear();

rand_vec(v);
tbb::parallel_sort(tbb::blocked_range<std::vector<int>::iterator>(v.begin(), v.end()), std::greater<int>());
for(itor a=v.begin(); a<v.end()-1; ++a) ASSERT(*a >= *(a+1), "v not sorted");
v.clear();

int arr[elements];
for(int i=0; i<elements; ++i) arr[i] = rand()%(elements*10);
tbb::parallel_sort(arr);
for(int i=0; i<elements-1; ++i) ASSERT(arr[i] <= arr[i+1], "arr not sorted");
}

#include <cstdio>
#include "harness_cpu.h"

int TestMain () {
if( MinThread<1 ) {
REPORT("Usage: number of threads must be positive\n");
exit(1);
}
for( int p=MinThread; p<=MaxThread; ++p ) {
if( p>0 ) {
tbb::task_scheduler_init init( p );
current_p = p;
Flog();
range_sort_test();

TestCPUUserTime(p);
}
}
return Harness::Done;
}

