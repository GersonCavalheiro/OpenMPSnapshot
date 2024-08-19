
#ifndef RAPIDJSON_RAPIDJSON_H_
#define RAPIDJSON_RAPIDJSON_H_






#include <cstdlib>  
#include <cstring>  




#ifndef RAPIDJSON_NAMESPACE
#define RAPIDJSON_NAMESPACE rapidjson
#endif
#ifndef RAPIDJSON_NAMESPACE_BEGIN
#define RAPIDJSON_NAMESPACE_BEGIN namespace RAPIDJSON_NAMESPACE {
#endif
#ifndef RAPIDJSON_NAMESPACE_END
#define RAPIDJSON_NAMESPACE_END }
#endif



#ifndef RAPIDJSON_NO_INT64DEFINE
#ifdef _MSC_VER
#include "msinttypes/stdint.h"
#include "msinttypes/inttypes.h"
#else
#include <stdint.h>
#include <inttypes.h>
#endif
#ifdef RAPIDJSON_DOXYGEN_RUNNING
#define RAPIDJSON_NO_INT64DEFINE
#endif
#endif 


#ifndef RAPIDJSON_FORCEINLINE
#ifdef _MSC_VER
#define RAPIDJSON_FORCEINLINE __forceinline
#elif defined(__GNUC__) && __GNUC__ >= 4
#define RAPIDJSON_FORCEINLINE __attribute__((always_inline))
#else
#define RAPIDJSON_FORCEINLINE
#endif
#endif 

#define RAPIDJSON_LITTLEENDIAN  0   
#define RAPIDJSON_BIGENDIAN     1   


#ifndef RAPIDJSON_ENDIAN
#  ifdef __BYTE_ORDER__
#    if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#      define RAPIDJSON_ENDIAN RAPIDJSON_LITTLEENDIAN
#    elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#      define RAPIDJSON_ENDIAN RAPIDJSON_BIGENDIAN
#    else
#      error Unknown machine endianess detected. User needs to define RAPIDJSON_ENDIAN.
#    endif 
#  elif defined(__GLIBC__)
#    include <endian.h>
#    if (__BYTE_ORDER == __LITTLE_ENDIAN)
#      define RAPIDJSON_ENDIAN RAPIDJSON_LITTLEENDIAN
#    elif (__BYTE_ORDER == __BIG_ENDIAN)
#      define RAPIDJSON_ENDIAN RAPIDJSON_BIGENDIAN
#    else
#      error Unknown machine endianess detected. User needs to define RAPIDJSON_ENDIAN.
#   endif 
#  elif defined(_LITTLE_ENDIAN) && !defined(_BIG_ENDIAN)
#    define RAPIDJSON_ENDIAN RAPIDJSON_LITTLEENDIAN
#  elif defined(_BIG_ENDIAN) && !defined(_LITTLE_ENDIAN)
#    define RAPIDJSON_ENDIAN RAPIDJSON_BIGENDIAN
#  elif defined(__sparc) || defined(__sparc__) || defined(_POWER) || defined(__powerpc__) || defined(__ppc__) || defined(__hpux) || defined(__hppa) || defined(_MIPSEB) || defined(_POWER) || defined(__s390__)
#    define RAPIDJSON_ENDIAN RAPIDJSON_BIGENDIAN
#  elif defined(__i386__) || defined(__alpha__) || defined(__ia64) || defined(__ia64__) || defined(_M_IX86) || defined(_M_IA64) || defined(_M_ALPHA) || defined(__amd64) || defined(__amd64__) || defined(_M_AMD64) || defined(__x86_64) || defined(__x86_64__) || defined(_M_X64) || defined(__bfin__)
#    define RAPIDJSON_ENDIAN RAPIDJSON_LITTLEENDIAN
#  elif defined(RAPIDJSON_DOXYGEN_RUNNING)
#    define RAPIDJSON_ENDIAN
#  else
#    error Unknown machine endianess detected. User needs to define RAPIDJSON_ENDIAN.   
#  endif
#endif 


#ifndef RAPIDJSON_64BIT
#if defined(__LP64__) || defined(_WIN64)
#define RAPIDJSON_64BIT 1
#else
#define RAPIDJSON_64BIT 0
#endif
#endif 



#ifndef RAPIDJSON_ALIGN
#define RAPIDJSON_ALIGN(x) ((x + 3u) & ~3u)
#endif



#ifndef RAPIDJSON_UINT64_C2
#define RAPIDJSON_UINT64_C2(high32, low32) ((static_cast<uint64_t>(high32) << 32) | static_cast<uint64_t>(low32))
#endif



#if defined(RAPIDJSON_SSE2) || defined(RAPIDJSON_SSE42) \
|| defined(RAPIDJSON_DOXYGEN_RUNNING)
#define RAPIDJSON_SIMD
#endif


#ifndef RAPIDJSON_NO_SIZETYPEDEFINE

#ifdef RAPIDJSON_DOXYGEN_RUNNING
#define RAPIDJSON_NO_SIZETYPEDEFINE
#endif
RAPIDJSON_NAMESPACE_BEGIN

typedef unsigned SizeType;
RAPIDJSON_NAMESPACE_END
#endif

RAPIDJSON_NAMESPACE_BEGIN
using std::size_t;
RAPIDJSON_NAMESPACE_END



#ifndef RAPIDJSON_ASSERT
#include <cassert>
#define RAPIDJSON_ASSERT(x) assert(x)
#endif 


#ifndef RAPIDJSON_STATIC_ASSERT
RAPIDJSON_NAMESPACE_BEGIN
template <bool x> struct STATIC_ASSERTION_FAILURE;
template <> struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };
template<int x> struct StaticAssertTest {};
RAPIDJSON_NAMESPACE_END

#define RAPIDJSON_JOIN(X, Y) RAPIDJSON_DO_JOIN(X, Y)
#define RAPIDJSON_DO_JOIN(X, Y) RAPIDJSON_DO_JOIN2(X, Y)
#define RAPIDJSON_DO_JOIN2(X, Y) X##Y

#if defined(__GNUC__)
#define RAPIDJSON_STATIC_ASSERT_UNUSED_ATTRIBUTE __attribute__((unused))
#else
#define RAPIDJSON_STATIC_ASSERT_UNUSED_ATTRIBUTE 
#endif


#define RAPIDJSON_STATIC_ASSERT(x) \
typedef ::RAPIDJSON_NAMESPACE::StaticAssertTest< \
sizeof(::RAPIDJSON_NAMESPACE::STATIC_ASSERTION_FAILURE<bool(x) >)> \
RAPIDJSON_JOIN(StaticAssertTypedef, __LINE__) RAPIDJSON_STATIC_ASSERT_UNUSED_ATTRIBUTE
#endif



#define RAPIDJSON_MULTILINEMACRO_BEGIN do {  
#define RAPIDJSON_MULTILINEMACRO_END \
} while((void)0, 0)

#define RAPIDJSON_VERSION_CODE(x,y,z) \
(((x)*100000) + ((y)*100) + (z))

#define RAPIDJSON_STRINGIFY(x) RAPIDJSON_DO_STRINGIFY(x)
#define RAPIDJSON_DO_STRINGIFY(x) #x


#if defined(__GNUC__)
#define RAPIDJSON_GNUC \
RAPIDJSON_VERSION_CODE(__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__)
#endif

#if defined(__clang__) || (defined(RAPIDJSON_GNUC) && RAPIDJSON_GNUC >= RAPIDJSON_VERSION_CODE(4,2,0))

#define RAPIDJSON_PRAGMA(x) _Pragma(RAPIDJSON_STRINGIFY(x))
#define RAPIDJSON_DIAG_PRAGMA(x) RAPIDJSON_PRAGMA(GCC diagnostic x)
#define RAPIDJSON_DIAG_OFF(x) \
RAPIDJSON_DIAG_PRAGMA(ignored RAPIDJSON_STRINGIFY(RAPIDJSON_JOIN(-W,x)))

#if defined(__clang__) || (defined(RAPIDJSON_GNUC) && RAPIDJSON_GNUC >= RAPIDJSON_VERSION_CODE(4,6,0))
#define RAPIDJSON_DIAG_PUSH RAPIDJSON_DIAG_PRAGMA(push)
#define RAPIDJSON_DIAG_POP  RAPIDJSON_DIAG_PRAGMA(pop)
#else 
#define RAPIDJSON_DIAG_PUSH 
#define RAPIDJSON_DIAG_POP 
#endif

#elif defined(_MSC_VER)

#define RAPIDJSON_PRAGMA(x) __pragma(x)
#define RAPIDJSON_DIAG_PRAGMA(x) RAPIDJSON_PRAGMA(warning(x))

#define RAPIDJSON_DIAG_OFF(x) RAPIDJSON_DIAG_PRAGMA(disable: x)
#define RAPIDJSON_DIAG_PUSH RAPIDJSON_DIAG_PRAGMA(push)
#define RAPIDJSON_DIAG_POP  RAPIDJSON_DIAG_PRAGMA(pop)

#else

#define RAPIDJSON_DIAG_OFF(x) 
#define RAPIDJSON_DIAG_PUSH   
#define RAPIDJSON_DIAG_POP    

#endif 


#ifndef RAPIDJSON_HAS_CXX11_RVALUE_REFS
#if defined(__clang__)
#define RAPIDJSON_HAS_CXX11_RVALUE_REFS __has_feature(cxx_rvalue_references)
#elif (defined(RAPIDJSON_GNUC) && (RAPIDJSON_GNUC >= RAPIDJSON_VERSION_CODE(4,3,0)) && defined(__GXX_EXPERIMENTAL_CXX0X__)) || \
(defined(_MSC_VER) && _MSC_VER >= 1600)

#define RAPIDJSON_HAS_CXX11_RVALUE_REFS 1
#else
#define RAPIDJSON_HAS_CXX11_RVALUE_REFS 0
#endif
#endif 

#ifndef RAPIDJSON_HAS_CXX11_NOEXCEPT
#if defined(__clang__)
#define RAPIDJSON_HAS_CXX11_NOEXCEPT __has_feature(cxx_noexcept)
#elif (defined(RAPIDJSON_GNUC) && (RAPIDJSON_GNUC >= RAPIDJSON_VERSION_CODE(4,6,0)) && defined(__GXX_EXPERIMENTAL_CXX0X__))
#define RAPIDJSON_HAS_CXX11_NOEXCEPT 1
#else
#define RAPIDJSON_HAS_CXX11_NOEXCEPT 0
#endif
#endif
#if RAPIDJSON_HAS_CXX11_NOEXCEPT
#define RAPIDJSON_NOEXCEPT noexcept
#else
#define RAPIDJSON_NOEXCEPT 
#endif 

#ifndef RAPIDJSON_HAS_CXX11_TYPETRAITS
#define RAPIDJSON_HAS_CXX11_TYPETRAITS 0
#endif



#ifndef RAPIDJSON_NEW
#define RAPIDJSON_NEW(x) new x
#endif
#ifndef RAPIDJSON_DELETE
#define RAPIDJSON_DELETE(x) delete x
#endif


#include "allocators.h"
#include "encodings.h"


RAPIDJSON_NAMESPACE_BEGIN





template<typename Stream>
struct StreamTraits {

enum { copyOptimization = 0 };
};

template<typename Stream, typename Ch>
inline void PutN(Stream& stream, Ch c, size_t n) {
for (size_t i = 0; i < n; i++)
stream.Put(c);
}



template <typename Encoding>
struct GenericStringStream {
typedef typename Encoding::Ch Ch;

GenericStringStream(const Ch *src) : src_(src), head_(src) {}

Ch Peek() const { return *src_; }
Ch Take() { return *src_++; }
size_t Tell() const { return static_cast<size_t>(src_ - head_); }

Ch* PutBegin() { RAPIDJSON_ASSERT(false); return 0; }
void Put(Ch) { RAPIDJSON_ASSERT(false); }
void Flush() { RAPIDJSON_ASSERT(false); }
size_t PutEnd(Ch*) { RAPIDJSON_ASSERT(false); return 0; }

const Ch* src_;     
const Ch* head_;    
};

template <typename Encoding>
struct StreamTraits<GenericStringStream<Encoding> > {
enum { copyOptimization = 1 };
};

typedef GenericStringStream<UTF8<> > StringStream;



template <typename Encoding>
struct GenericInsituStringStream {
typedef typename Encoding::Ch Ch;

GenericInsituStringStream(Ch *src) : src_(src), dst_(0), head_(src) {}

Ch Peek() { return *src_; }
Ch Take() { return *src_++; }
size_t Tell() { return static_cast<size_t>(src_ - head_); }

void Put(Ch c) { RAPIDJSON_ASSERT(dst_ != 0); *dst_++ = c; }

Ch* PutBegin() { return dst_ = src_; }
size_t PutEnd(Ch* begin) { return static_cast<size_t>(dst_ - begin); }
void Flush() {}

Ch* Push(size_t count) { Ch* begin = dst_; dst_ += count; return begin; }
void Pop(size_t count) { dst_ -= count; }

Ch* src_;
Ch* dst_;
Ch* head_;
};

template <typename Encoding>
struct StreamTraits<GenericInsituStringStream<Encoding> > {
enum { copyOptimization = 1 };
};

typedef GenericInsituStringStream<UTF8<> > InsituStringStream;


enum Type {
kNullType = 0,      
kFalseType = 1,     
kTrueType = 2,      
kObjectType = 3,    
kArrayType = 4,     
kStringType = 5,    
kNumberType = 6     
};

RAPIDJSON_NAMESPACE_END

#endif 
