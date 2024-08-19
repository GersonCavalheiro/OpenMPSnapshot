

#include "tbb/tbb_config.h"

#if __TBB_GCC_STRICT_ALIASING_BROKEN
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif


template<typename T>
class Minimal {
Minimal();
Minimal( Minimal& min );
~Minimal();
void operator=( const Minimal& );
T pad;
template<typename U>
friend void AssignToCheckAlignment( Minimal<U>& dst, const Minimal<U>& src ) ;
};

template<typename T>
void AssignToCheckAlignment( Minimal<T>& dst, const Minimal<T>& src ) {
dst.pad = src.pad;
}

#include "tbb/aligned_space.h"
#include "harness_assert.h"

static bool SpaceWasted;

template<typename U, size_t N>
void TestAlignedSpaceN() {
typedef Minimal<U> T;
struct {
char pad;
tbb::aligned_space<T ,N> space;
} x;
AssertSameType( static_cast< T *>(0), x.space.begin() );
AssertSameType( static_cast< T *>(0), x.space.end() );
ASSERT( reinterpret_cast<void *>(x.space.begin())==reinterpret_cast< void *>(&x.space), NULL );
ASSERT( x.space.end()-x.space.begin()==N, NULL );
ASSERT( reinterpret_cast<void *>(x.space.begin())>=reinterpret_cast< void *>(&x.space), NULL );
ASSERT( x.space.end()<=reinterpret_cast< T *>(&x.space+1), NULL );
SpaceWasted |= sizeof(x.space)!=sizeof(T)*N;
for( size_t k=1; k<N; ++k )
AssignToCheckAlignment( x.space.begin()[k-1], x.space.begin()[k] );
}

#define HARNESS_NO_PARSE_COMMAND_LINE 1
#include "harness.h"

#include <typeinfo>
template<typename T>
void PrintSpaceWastingWarning() {
REPORT( "Consider rewriting aligned_space<%s,N> to waste less space\n", typeid(T).name() );
}

#if ( __ANDROID__ && __clang__ && _LIBCPP_VERSION && __TBB_x86_64 )
template<>
void PrintSpaceWastingWarning<long double>() {
REPORT( "Consider rewriting aligned_space<ld,N> to waste less space\n" );
}
#endif

template<typename T>
void TestAlignedSpace() {
SpaceWasted = false;
TestAlignedSpaceN<T,1>();
TestAlignedSpaceN<T,2>();
TestAlignedSpaceN<T,3>();
TestAlignedSpaceN<T,4>();
TestAlignedSpaceN<T,5>();
TestAlignedSpaceN<T,6>();
TestAlignedSpaceN<T,7>();
TestAlignedSpaceN<T,8>();
if( SpaceWasted )
PrintSpaceWastingWarning<T>();
}

#include "harness_m128.h"

int TestMain () {
TestAlignedSpace<char>();
TestAlignedSpace<short>();
TestAlignedSpace<int>();
TestAlignedSpace<float>();
TestAlignedSpace<double>();
TestAlignedSpace<long double>();
TestAlignedSpace<size_t>();
#if HAVE_m128
TestAlignedSpace<__m128>();
#endif
#if HAVE_m256
if (have_AVX()) TestAlignedSpace<__m256>();
#endif
return Harness::Done;
}
