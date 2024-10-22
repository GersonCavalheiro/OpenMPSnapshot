


#include "harness.h"
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> 
#endif

template<typename A>
struct is_zero_filling {
static const bool value = false;
};

int NumberOfFoo;

template<typename T, size_t N>
struct Foo {
T foo_array[N];
Foo() {
zero_fill<T>(foo_array, N);
++NumberOfFoo;
}
Foo( const Foo& x ) {
*this = x;
}
Foo& operator=( const Foo& x ) {
for (size_t i = 0; i < N; i++)
foo_array[i] = x.foo_array[i];
++NumberOfFoo;
return *this;
}

~Foo() {
--NumberOfFoo;
}
};

inline char PseudoRandomValue( size_t j, size_t k ) {
return char(j*3 ^ j>>4 ^ k);
}

#if __APPLE__
#include <fcntl.h>
#include <unistd.h>

class DisableStderr {
int stderrCopy;
static void dupToStderrAndClose(int fd) {
int ret = dup2(fd, STDERR_FILENO); 
ASSERT(ret != -1, NULL);
ret = close(fd);
ASSERT(ret != -1, NULL);
}
public:
DisableStderr() {
int devNull = open("/dev/null", O_WRONLY);
ASSERT(devNull != -1, NULL);
stderrCopy = dup(STDERR_FILENO);
ASSERT(stderrCopy != -1, NULL);
dupToStderrAndClose(devNull);
}
~DisableStderr() {
dupToStderrAndClose(stderrCopy);
}
};
#endif

template<typename T, typename A>
void TestBasic( A& a ) {
T x;
const T cx = T();

typename A::pointer px = &x;
typename A::const_pointer pcx = &cx;

typename A::reference rx = x;
ASSERT( &rx==&x, NULL );

typename A::const_reference rcx = cx;
ASSERT( &rcx==&cx, NULL );

typename A::value_type v = x;

typename A::size_type size;
size = 0;
--size;
ASSERT( size>0, "not an unsigned integral type?" );

typename A::difference_type difference;
difference = 0;
--difference;
ASSERT( difference<0, "not an signed integral type?" );


ASSERT( a.address(rx)==px, NULL );

ASSERT( a.address(rcx)==pcx, NULL );

typename A::pointer array[100];
size_t sizeof_T = sizeof(T);
for( size_t k=0; k<100; ++k ) {
array[k] = k&1 ? a.allocate(k,array[0]) : a.allocate(k);
char* s = reinterpret_cast<char*>(reinterpret_cast<void*>(array[k]));
for( size_t j=0; j<k*sizeof_T; ++j )
s[j] = PseudoRandomValue(j,k);
}

typename A::pointer a_ptr;
const void * const_hint = NULL;
a_ptr = a.allocate (1, const_hint);
a.deallocate(a_ptr, 1);

for( size_t k=0; k<100; ++k ) {
char* s = reinterpret_cast<char*>(reinterpret_cast<void*>(array[k]));
for( size_t j=0; j<k*sizeof_T; ++j )
ASSERT( s[j] == PseudoRandomValue(j,k), NULL );
a.deallocate(array[k],k);
}

AssertSameType( a.max_size(), typename A::size_type(0) );
ASSERT( a.max_size()*typename A::size_type(sizeof(T))>=a.max_size(), "max_size larger than reasonable" );

int n = NumberOfFoo;
typename A::pointer p = a.allocate(1);
a.construct( p, cx );
ASSERT( NumberOfFoo==n+1, "constructor for Foo not called?" );

a.destroy( p );
ASSERT( NumberOfFoo==n, "destructor for Foo not called?" );
a.deallocate(p,1);

#if TBB_USE_EXCEPTIONS
volatile size_t too_big = (~size_t(0) - 1024*1024)/sizeof(T);
bool exception_caught = false;
typename A::pointer p1 = NULL;
try {
#if __APPLE__
DisableStderr disableStderr;
#endif
p1 = a.allocate(too_big);
} catch ( std::bad_alloc& ) {
exception_caught = true;
}
ASSERT( exception_caught, "allocate expected to throw bad_alloc" );
a.deallocate(p1, too_big);
#endif 

#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
{
typedef typename A:: template rebind<std::pair<typename A::value_type, typename A::value_type> >::other pair_allocator_type;
pair_allocator_type pair_allocator(a);
int NumberOfFooBeforeConstruct= NumberOfFoo;
typename pair_allocator_type::pointer pair_pointer = pair_allocator.allocate(1);
pair_allocator.construct( pair_pointer, cx, cx);
ASSERT( NumberOfFoo==NumberOfFooBeforeConstruct+2, "constructor for Foo not called appropriate number of times?" );

pair_allocator.destroy( pair_pointer );
ASSERT( NumberOfFoo==NumberOfFooBeforeConstruct, "destructor for Foo not called appropriate number of times?" );
pair_allocator.deallocate(pair_pointer,1);
}
#endif

}

#include "tbb/blocked_range.h"

#if _MSC_VER && !defined(__INTEL_COMPILER)
#pragma warning (disable: 4127)
#endif

template<typename A>
struct Body: NoAssign {
static const size_t max_k = 100000;
A &a;
Body(A &a_) : a(a_) {}
void check_allocate( typename A::pointer array[], size_t i, size_t t ) const
{
ASSERT(array[i] == 0, NULL);
size_t size = i * (i&3);
array[i] = i&1 ? a.allocate(size, array[i>>3]) : a.allocate(size);
ASSERT(array[i] != 0, "allocator returned null");
char* s = reinterpret_cast<char*>(reinterpret_cast<void*>(array[i]));
for( size_t j=0; j<size*sizeof(typename A::value_type); ++j ) {
if(is_zero_filling<typename A::template rebind<void>::other>::value)
ASSERT( !s[j], NULL);
s[j] = PseudoRandomValue(i, t);
}
}

void check_deallocate( typename A::pointer array[], size_t i, size_t t ) const
{
ASSERT(array[i] != 0, NULL);
size_t size = i * (i&3);
char* s = reinterpret_cast<char*>(reinterpret_cast<void*>(array[i]));
for( size_t j=0; j<size*sizeof(typename A::value_type); ++j )
ASSERT( s[j] == PseudoRandomValue(i, t), "Thread safety test failed" );
a.deallocate(array[i], size);
array[i] = 0;
}

void operator()( size_t thread_id ) const {
typename A::pointer array[256];

for( size_t k=0; k<256; ++k )
array[k] = 0;
for( size_t k=0; k<max_k; ++k ) {
size_t i = static_cast<unsigned char>(PseudoRandomValue(k,thread_id));
if(!array[i]) check_allocate(array, i, thread_id);
else check_deallocate(array, i, thread_id);
}
for( size_t k=0; k<256; ++k )
if(array[k])
check_deallocate(array, k, thread_id);
}
};

template<typename U, typename A>
void Test(A &a) {
typename A::template rebind<U>::other b(a);
TestBasic<U>(b);
TestBasic<typename A::value_type>(a);

NativeParallelFor( 4, Body<A>(a) );
ASSERT( NumberOfFoo==0, "Allocate/deallocate count mismatched" );

ASSERT( a==b, NULL );
ASSERT( !(a!=b), NULL );
}

template<typename Allocator>
int TestMain(const Allocator &a = Allocator()) {
NumberOfFoo = 0;
typename Allocator::template rebind<Foo<char,1> >::other a1(a);
typename Allocator::template rebind<Foo<double,1> >::other a2(a);
Test<Foo<int,17> >( a1 );
Test<Foo<float,23> >( a2 );
return 0;
}
