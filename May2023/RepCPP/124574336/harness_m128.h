




#if (_MSC_VER>=1600)
#pragma warning (push)
#pragma warning (disable: 4752)
#endif

#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT && __TBB_GCC_WARNING_IGNORED_ATTRIBUTES_PRESENT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif


template<typename __Mvec>
class ClassWithVectorType {
static const int n = 16;
static const int F = sizeof(__Mvec)/sizeof(float);
__Mvec field[n];
void init( int start );
public:
ClassWithVectorType() {init(-n);}
ClassWithVectorType( int i ) {init(i);}
ClassWithVectorType( const ClassWithVectorType& src ) {
for( int i=0; i<n; ++i ) {
field[i] = src.field[i];
}
}
void operator=( const ClassWithVectorType& src ) {
__Mvec stack[n];
for( int i=0; i<n; ++i )
stack[i^5] = src.field[i];
for( int i=0; i<n; ++i )
field[i^5] = stack[i];
}
~ClassWithVectorType() {init(-2*n);}
friend bool operator==( const ClassWithVectorType& x, const ClassWithVectorType& y ) {
for( int i=0; i<F*n; ++i )
if( ((const float*)x.field)[i]!=((const float*)y.field)[i] )
return false;
return true;
}
friend bool operator!=( const ClassWithVectorType& x, const ClassWithVectorType& y ) {
return !(x==y);
}
};

template<typename __Mvec>
void ClassWithVectorType<__Mvec>::init( int start ) {
__Mvec stack[n];
for( int i=0; i<n; ++i ) {
__Mvec value[1];
for( int j=0; j<F; ++j )
((float*)value)[j] = float(n*start+F*i+j);
stack[i^5] = value[0];
}
for( int i=0; i<n; ++i )
field[i^5] = stack[i];
}

#if (__AVX__ || (_MSC_VER>=1600 && _M_X64)) && !defined(__sun)
#include <immintrin.h>
#define HAVE_m256 1
typedef ClassWithVectorType<__m256> ClassWithAVX;
#if _MSC_VER
#include <intrin.h> 
#endif
bool have_AVX() {
bool result = false;
const int avx_mask = 1<<28;
#if _MSC_VER || __INTEL_COMPILER
int info[4] = {0,0,0,0};
const int ECX = 2;
__cpuid(info, 1);
result = (info[ECX] & avx_mask)!=0;
#elif __GNUC__
int ECX;
__asm__( "cpuid"
: "=c"(ECX)
: "a" (1)
: "ebx", "edx" );
result = (ECX & avx_mask);
#endif
return result;
}
#endif 

#if (__SSE__ || _M_IX86_FP || _M_X64) && !defined(__sun)
#include <xmmintrin.h>
#define HAVE_m128 1
typedef ClassWithVectorType<__m128> ClassWithSSE;
#endif

#if __TBB_GCC_WARNING_SUPPRESSION_PRESENT && __TBB_GCC_WARNING_IGNORED_ATTRIBUTES_PRESENT
#pragma GCC diagnostic pop
#endif

#if (_MSC_VER>=1600)
#pragma warning (pop)
#endif
