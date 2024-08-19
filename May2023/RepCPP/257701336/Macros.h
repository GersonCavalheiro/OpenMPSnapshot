
#ifndef EIGEN_MACROS_H
#define EIGEN_MACROS_H


#define EIGEN_WORLD_VERSION 3
#define EIGEN_MAJOR_VERSION 3
#define EIGEN_MINOR_VERSION 90

#define EIGEN_VERSION_AT_LEAST(x,y,z) (EIGEN_WORLD_VERSION>x || (EIGEN_WORLD_VERSION>=x && \
(EIGEN_MAJOR_VERSION>y || (EIGEN_MAJOR_VERSION>=y && \
EIGEN_MINOR_VERSION>=z))))

#ifdef EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::RowMajor
#else
#define EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION Eigen::ColMajor
#endif

#ifndef EIGEN_DEFAULT_DENSE_INDEX_TYPE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE std::ptrdiff_t
#endif

#ifndef EIGEN_MAX_CPP_VER
#define EIGEN_MAX_CPP_VER 99
#endif


#ifndef EIGEN_FAST_MATH
#define EIGEN_FAST_MATH 1
#endif

#ifndef EIGEN_STACK_ALLOCATION_LIMIT
#define EIGEN_STACK_ALLOCATION_LIMIT 131072
#endif


#ifdef __GNUC__
#define EIGEN_COMP_GNUC (__GNUC__*10+__GNUC_MINOR__)
#else
#define EIGEN_COMP_GNUC 0
#endif

#if defined(__clang__)
#define EIGEN_COMP_CLANG (__clang_major__*100+__clang_minor__)
#else
#define EIGEN_COMP_CLANG 0
#endif


#if defined(__llvm__)
#define EIGEN_COMP_LLVM 1
#else
#define EIGEN_COMP_LLVM 0
#endif

#if defined(__INTEL_COMPILER)
#define EIGEN_COMP_ICC __INTEL_COMPILER
#else
#define EIGEN_COMP_ICC 0
#endif

#if defined(__MINGW32__)
#define EIGEN_COMP_MINGW 1
#else
#define EIGEN_COMP_MINGW 0
#endif

#if defined(__SUNPRO_CC)
#define EIGEN_COMP_SUNCC 1
#else
#define EIGEN_COMP_SUNCC 0
#endif

#if defined(_MSC_VER)
#define EIGEN_COMP_MSVC _MSC_VER
#else
#define EIGEN_COMP_MSVC 0
#endif


#if EIGEN_COMP_MSVC && !(EIGEN_COMP_ICC || EIGEN_COMP_LLVM || EIGEN_COMP_CLANG)
#define EIGEN_COMP_MSVC_STRICT _MSC_VER
#else
#define EIGEN_COMP_MSVC_STRICT 0
#endif

#if defined(__IBMCPP__) || defined(__xlc__)
#define EIGEN_COMP_IBM 1
#else
#define EIGEN_COMP_IBM 0
#endif

#if defined(__PGI)
#define EIGEN_COMP_PGI 1
#else
#define EIGEN_COMP_PGI 0
#endif

#if defined(__CC_ARM) || defined(__ARMCC_VERSION)
#define EIGEN_COMP_ARM 1
#else
#define EIGEN_COMP_ARM 0
#endif

#if defined(__EMSCRIPTEN__)
#define EIGEN_COMP_EMSCRIPTEN 1
#else
#define EIGEN_COMP_EMSCRIPTEN 0
#endif


#if EIGEN_COMP_GNUC && !(EIGEN_COMP_CLANG || EIGEN_COMP_ICC || EIGEN_COMP_MINGW || EIGEN_COMP_PGI || EIGEN_COMP_IBM || EIGEN_COMP_ARM || EIGEN_COMP_EMSCRIPTEN)
#define EIGEN_COMP_GNUC_STRICT 1
#else
#define EIGEN_COMP_GNUC_STRICT 0
#endif


#if EIGEN_COMP_GNUC
#define EIGEN_GNUC_AT_LEAST(x,y) ((__GNUC__==x && __GNUC_MINOR__>=y) || __GNUC__>x)
#define EIGEN_GNUC_AT_MOST(x,y)  ((__GNUC__==x && __GNUC_MINOR__<=y) || __GNUC__<x)
#define EIGEN_GNUC_AT(x,y)       ( __GNUC__==x && __GNUC_MINOR__==y )
#else
#define EIGEN_GNUC_AT_LEAST(x,y) 0
#define EIGEN_GNUC_AT_MOST(x,y)  0
#define EIGEN_GNUC_AT(x,y)       0
#endif

#if EIGEN_COMP_GNUC && (__GNUC__ <= 3)
#define EIGEN_GCC3_OR_OLDER 1
#else
#define EIGEN_GCC3_OR_OLDER 0
#endif





#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
#define EIGEN_ARCH_x86_64 1
#else
#define EIGEN_ARCH_x86_64 0
#endif

#if defined(__i386__) || defined(_M_IX86) || defined(_X86_) || defined(__i386)
#define EIGEN_ARCH_i386 1
#else
#define EIGEN_ARCH_i386 0
#endif

#if EIGEN_ARCH_x86_64 || EIGEN_ARCH_i386
#define EIGEN_ARCH_i386_OR_x86_64 1
#else
#define EIGEN_ARCH_i386_OR_x86_64 0
#endif

#if defined(__arm__)
#define EIGEN_ARCH_ARM 1
#else
#define EIGEN_ARCH_ARM 0
#endif

#if defined(__aarch64__)
#define EIGEN_ARCH_ARM64 1
#else
#define EIGEN_ARCH_ARM64 0
#endif

#if EIGEN_ARCH_ARM || EIGEN_ARCH_ARM64
#define EIGEN_ARCH_ARM_OR_ARM64 1
#else
#define EIGEN_ARCH_ARM_OR_ARM64 0
#endif

#if defined(__mips__) || defined(__mips)
#define EIGEN_ARCH_MIPS 1
#else
#define EIGEN_ARCH_MIPS 0
#endif

#if defined(__sparc__) || defined(__sparc)
#define EIGEN_ARCH_SPARC 1
#else
#define EIGEN_ARCH_SPARC 0
#endif

#if defined(__ia64__)
#define EIGEN_ARCH_IA64 1
#else
#define EIGEN_ARCH_IA64 0
#endif

#if defined(__powerpc__) || defined(__ppc__) || defined(_M_PPC)
#define EIGEN_ARCH_PPC 1
#else
#define EIGEN_ARCH_PPC 0
#endif




#if defined(__unix__) || defined(__unix)
#define EIGEN_OS_UNIX 1
#else
#define EIGEN_OS_UNIX 0
#endif

#if defined(__linux__)
#define EIGEN_OS_LINUX 1
#else
#define EIGEN_OS_LINUX 0
#endif

#if defined(__ANDROID__) || defined(ANDROID)
#define EIGEN_OS_ANDROID 1
#else
#define EIGEN_OS_ANDROID 0
#endif

#if defined(__gnu_linux__) && !(EIGEN_OS_ANDROID)
#define EIGEN_OS_GNULINUX 1
#else
#define EIGEN_OS_GNULINUX 0
#endif

#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__DragonFly__)
#define EIGEN_OS_BSD 1
#else
#define EIGEN_OS_BSD 0
#endif

#if defined(__APPLE__)
#define EIGEN_OS_MAC 1
#else
#define EIGEN_OS_MAC 0
#endif

#if defined(__QNX__)
#define EIGEN_OS_QNX 1
#else
#define EIGEN_OS_QNX 0
#endif

#if defined(_WIN32)
#define EIGEN_OS_WIN 1
#else
#define EIGEN_OS_WIN 0
#endif

#if defined(_WIN64)
#define EIGEN_OS_WIN64 1
#else
#define EIGEN_OS_WIN64 0
#endif

#if defined(_WIN32_WCE)
#define EIGEN_OS_WINCE 1
#else
#define EIGEN_OS_WINCE 0
#endif

#if defined(__CYGWIN__)
#define EIGEN_OS_CYGWIN 1
#else
#define EIGEN_OS_CYGWIN 0
#endif

#if EIGEN_OS_WIN && !( EIGEN_OS_WINCE || EIGEN_OS_CYGWIN )
#define EIGEN_OS_WIN_STRICT 1
#else
#define EIGEN_OS_WIN_STRICT 0
#endif

#if (defined(sun) || defined(__sun)) && !(defined(__SVR4) || defined(__svr4__))
#define EIGEN_OS_SUN 1
#else
#define EIGEN_OS_SUN 0
#endif

#if (defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__))
#define EIGEN_OS_SOLARIS 1
#else
#define EIGEN_OS_SOLARIS 0
#endif



#if defined(__NVCC__) && defined(__HIPCC__)
#error "NVCC as the target platform for HIPCC is currently not supported."
#endif

#if defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_CUDACC __CUDACC__
#endif

#if defined(__CUDA_ARCH__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_CUDA_ARCH __CUDA_ARCH__
#endif

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
#define EIGEN_CUDACC_VER  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#elif defined(__CUDACC_VER__)
#define EIGEN_CUDACC_VER __CUDACC_VER__
#else
#define EIGEN_CUDACC_VER 0
#endif

#if defined(__HIPCC__) && !defined(EIGEN_NO_HIP)
#define EIGEN_HIPCC __HIPCC__

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__)
#define EIGEN_HIP_DEVICE_COMPILE __HIP_DEVICE_COMPILE__
#endif
#endif


#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
#define EIGEN_GPUCC
#endif

#if defined(EIGEN_CUDA_ARCH) || defined(EIGEN_HIP_DEVICE_COMPILE)
#define EIGEN_GPU_COMPILE_PHASE
#endif


#if EIGEN_GNUC_AT_MOST(4,3) && !EIGEN_COMP_CLANG
#define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 0
#else
#define EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO 1
#endif

#ifdef __has_builtin
#  define EIGEN_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define EIGEN_HAS_BUILTIN(x) 0
#endif

#ifndef __has_feature
# define __has_feature(x) 0
#endif

#if !(   EIGEN_COMP_CLANG && (   (EIGEN_COMP_CLANG<309)                                                       \
|| (defined(__apple_build_version__) && (__apple_build_version__ < 9000000)))  \
|| EIGEN_COMP_GNUC_STRICT && EIGEN_COMP_GNUC<49)
#define EIGEN_HAS_STATIC_ARRAY_TEMPLATE 1
#else
#define EIGEN_HAS_STATIC_ARRAY_TEMPLATE 0
#endif

#if EIGEN_MAX_CPP_VER>=11 && (defined(__cplusplus) && (__cplusplus >= 201103L) || EIGEN_COMP_MSVC >= 1900)
#define EIGEN_HAS_CXX11 1
#else
#define EIGEN_HAS_CXX11 0
#endif

#if EIGEN_MAX_CPP_VER>=14 && (defined(__cplusplus) && (__cplusplus > 201103L) || EIGEN_COMP_MSVC >= 1910)
#define EIGEN_HAS_CXX14 1
#else
#define EIGEN_HAS_CXX14 0
#endif

#ifndef EIGEN_HAS_RVALUE_REFERENCES
#if EIGEN_MAX_CPP_VER>=11 && \
(__has_feature(cxx_rvalue_references) || \
(defined(__cplusplus) && __cplusplus >= 201103L) || \
(EIGEN_COMP_MSVC >= 1600))
#define EIGEN_HAS_RVALUE_REFERENCES 1
#else
#define EIGEN_HAS_RVALUE_REFERENCES 0
#endif
#endif

#include <cmath>
#ifndef EIGEN_HAS_C99_MATH
#if EIGEN_MAX_CPP_VER>=11 && \
((defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901))       \
|| (defined(__GNUC__) && defined(_GLIBCXX_USE_C99)) \
|| (defined(_LIBCPP_VERSION) && !defined(_MSC_VER)) \
|| (EIGEN_COMP_MSVC >= 1900) || defined(__SYCL_DEVICE_ONLY__))
#define EIGEN_HAS_C99_MATH 1
#else
#define EIGEN_HAS_C99_MATH 0
#endif
#endif

#ifndef EIGEN_HAS_STD_RESULT_OF
#if EIGEN_MAX_CPP_VER>=11 && \
(__has_feature(cxx_lambdas) || (defined(__cplusplus) && __cplusplus >= 201103L) || EIGEN_COMP_MSVC >= 1900)
#define EIGEN_HAS_STD_RESULT_OF 1
#else
#define EIGEN_HAS_STD_RESULT_OF 0
#endif
#endif

#ifndef EIGEN_HAS_TYPE_TRAITS
#if EIGEN_MAX_CPP_VER>=11 && (EIGEN_HAS_CXX11 || EIGEN_COMP_MSVC >= 1700) \
&& ((!EIGEN_COMP_GNUC_STRICT) || EIGEN_GNUC_AT_LEAST(5, 1)) \
&& ((!defined(__GLIBCXX__))   || __GLIBCXX__ > 20150626)
#define EIGEN_HAS_TYPE_TRAITS 1
#define EIGEN_INCLUDE_TYPE_TRAITS
#else
#define EIGEN_HAS_TYPE_TRAITS 0
#endif
#endif

#ifndef EIGEN_HAS_VARIADIC_TEMPLATES
#if EIGEN_MAX_CPP_VER>=11 && (__cplusplus > 199711L || EIGEN_COMP_MSVC >= 1900) \
&& (!defined(__NVCC__) || !EIGEN_ARCH_ARM_OR_ARM64 || (EIGEN_CUDACC_VER >= 80000) )
#define EIGEN_HAS_VARIADIC_TEMPLATES 1
#elif  EIGEN_MAX_CPP_VER>=11 && (__cplusplus > 199711L || EIGEN_COMP_MSVC >= 1900) && defined(__SYCL_DEVICE_ONLY__)
#define EIGEN_HAS_VARIADIC_TEMPLATES 1
#else
#define EIGEN_HAS_VARIADIC_TEMPLATES 0
#endif
#endif

#ifndef EIGEN_HAS_CONSTEXPR
#if defined(EIGEN_CUDACC)
#if EIGEN_MAX_CPP_VER>=14 && (__cplusplus > 199711L && (EIGEN_COMP_CLANG || EIGEN_CUDACC_VER >= 70500))
#define EIGEN_HAS_CONSTEXPR 1
#endif
#elif EIGEN_MAX_CPP_VER>=14 && (__has_feature(cxx_relaxed_constexpr) || (defined(__cplusplus) && __cplusplus >= 201402L) || \
(EIGEN_GNUC_AT_LEAST(4,8) && (__cplusplus > 199711L)) || \
(EIGEN_COMP_CLANG >= 306 && (__cplusplus > 199711L)))
#define EIGEN_HAS_CONSTEXPR 1
#endif

#ifndef EIGEN_HAS_CONSTEXPR
#define EIGEN_HAS_CONSTEXPR 0
#endif

#endif 

#ifndef EIGEN_HAS_CXX11_MATH
#if EIGEN_MAX_CPP_VER>=11 && ((__cplusplus > 201103L) || (__cplusplus >= 201103L) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG || EIGEN_COMP_MSVC || EIGEN_COMP_ICC)  \
&& (EIGEN_ARCH_i386_OR_x86_64) && (EIGEN_OS_GNULINUX || EIGEN_OS_WIN_STRICT || EIGEN_OS_MAC))
#define EIGEN_HAS_CXX11_MATH 1
#else
#define EIGEN_HAS_CXX11_MATH 0
#endif
#endif

#ifndef EIGEN_HAS_CXX11_CONTAINERS
#if    EIGEN_MAX_CPP_VER>=11 && \
((__cplusplus > 201103L) \
|| ((__cplusplus >= 201103L) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG || EIGEN_COMP_ICC>=1400)) \
|| EIGEN_COMP_MSVC >= 1900)
#define EIGEN_HAS_CXX11_CONTAINERS 1
#else
#define EIGEN_HAS_CXX11_CONTAINERS 0
#endif
#endif

#ifndef EIGEN_HAS_CXX11_NOEXCEPT
#if    EIGEN_MAX_CPP_VER>=11 && \
(__has_feature(cxx_noexcept) \
|| (__cplusplus > 201103L) \
|| ((__cplusplus >= 201103L) && (EIGEN_COMP_GNUC_STRICT || EIGEN_COMP_CLANG || EIGEN_COMP_ICC>=1400)) \
|| EIGEN_COMP_MSVC >= 1900)
#define EIGEN_HAS_CXX11_NOEXCEPT 1
#else
#define EIGEN_HAS_CXX11_NOEXCEPT 0
#endif
#endif

#ifndef EIGEN_HAS_CXX11_ATOMIC
#if    EIGEN_MAX_CPP_VER>=11 && \
(__has_feature(cxx_atomic) \
|| (__cplusplus > 201103L) \
|| ((__cplusplus >= 201103L) && (EIGEN_COMP_MSVC==0 || EIGEN_COMP_MSVC >= 1700)))
#define EIGEN_HAS_CXX11_ATOMIC 1
#else
#define EIGEN_HAS_CXX11_ATOMIC 0
#endif
#endif

#ifndef EIGEN_HAS_CXX11_OVERRIDE_FINAL
#if    EIGEN_MAX_CPP_VER>=11 && \
(__cplusplus >= 201103L || EIGEN_COMP_MSVC >= 1700)
#define EIGEN_HAS_CXX11_OVERRIDE_FINAL 1
#else
#define EIGEN_HAS_CXX11_OVERRIDE_FINAL 0
#endif
#endif

#if defined(EIGEN_CUDACC) && EIGEN_HAS_CONSTEXPR
#if defined(__NVCC__)
#ifdef __CUDACC_RELAXED_CONSTEXPR__
#define EIGEN_CONSTEXPR_ARE_DEVICE_FUNC
#endif
#elif defined(__clang__) && defined(__CUDA__) && __has_feature(cxx_relaxed_constexpr)
#define EIGEN_CONSTEXPR_ARE_DEVICE_FUNC
#endif
#endif



#define EIGEN_NOT_A_MACRO

#define EIGEN_DEBUG_VAR(x) std::cerr << #x << " = " << x << std::endl;

#define EIGEN_CAT2(a,b) a ## b
#define EIGEN_CAT(a,b) EIGEN_CAT2(a,b)

#define EIGEN_COMMA ,

#define EIGEN_MAKESTRING2(a) #a
#define EIGEN_MAKESTRING(a) EIGEN_MAKESTRING2(a)

#ifndef EIGEN_STRONG_INLINE
#if EIGEN_COMP_MSVC || EIGEN_COMP_ICC
#define EIGEN_STRONG_INLINE __forceinline
#else
#define EIGEN_STRONG_INLINE inline
#endif
#endif

#if EIGEN_GNUC_AT_LEAST(4,2)
#define EIGEN_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define EIGEN_ALWAYS_INLINE EIGEN_STRONG_INLINE
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_DONT_INLINE __attribute__((noinline))
#elif EIGEN_COMP_MSVC
#define EIGEN_DONT_INLINE __declspec(noinline)
#else
#define EIGEN_DONT_INLINE
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_PERMISSIVE_EXPR __extension__
#else
#define EIGEN_PERMISSIVE_EXPR
#endif


#if defined(EIGEN_CUDACC) || defined(__SYCL_DEVICE_ONLY__) || defined(EIGEN_HIPCC)
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

#ifdef EIGEN_INTERNAL_DEBUGGING
#undef EIGEN_INTERNAL_DEBUGGING
#endif

#ifdef EIGEN_EXCEPTIONS
#undef EIGEN_EXCEPTIONS
#endif
#endif

#ifdef EIGEN_GPUCC
#define EIGEN_DEVICE_FUNC __host__ __device__
#else
#define EIGEN_DEVICE_FUNC
#endif


#define EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_DEVICE_FUNC
#define EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_DEVICE_FUNC inline

#ifdef NDEBUG
# ifndef EIGEN_NO_DEBUG
#  define EIGEN_NO_DEBUG
# endif
#endif

#ifdef EIGEN_NO_DEBUG
#define eigen_plain_assert(x)
#else
#if EIGEN_SAFE_TO_USE_STANDARD_ASSERT_MACRO
namespace Eigen {
namespace internal {
inline bool copy_bool(bool b) { return b; }
}
}
#define eigen_plain_assert(x) assert(x)
#else
#include <cstdlib>   
#include <iostream>  

namespace Eigen {
namespace internal {
namespace {
EIGEN_DONT_INLINE bool copy_bool(bool b) { return b; }
}
inline void assert_fail(const char *condition, const char *function, const char *file, int line)
{
std::cerr << "assertion failed: " << condition << " in function " << function << " at " << file << ":" << line << std::endl;
abort();
}
}
}
#define eigen_plain_assert(x) \
do { \
if(!Eigen::internal::copy_bool(x)) \
Eigen::internal::assert_fail(EIGEN_MAKESTRING(x), __PRETTY_FUNCTION__, __FILE__, __LINE__); \
} while(false)
#endif
#endif

#ifndef eigen_assert
#define eigen_assert(x) eigen_plain_assert(x)
#endif

#ifdef EIGEN_INTERNAL_DEBUGGING
#define eigen_internal_assert(x) eigen_assert(x)
#else
#define eigen_internal_assert(x)
#endif

#ifdef EIGEN_NO_DEBUG
#define EIGEN_ONLY_USED_FOR_DEBUG(x) EIGEN_UNUSED_VARIABLE(x)
#else
#define EIGEN_ONLY_USED_FOR_DEBUG(x)
#endif

#ifndef EIGEN_NO_DEPRECATED_WARNING
#if EIGEN_COMP_GNUC
#define EIGEN_DEPRECATED __attribute__((deprecated))
#elif EIGEN_COMP_MSVC
#define EIGEN_DEPRECATED __declspec(deprecated)
#else
#define EIGEN_DEPRECATED
#endif
#else
#define EIGEN_DEPRECATED
#endif

#if EIGEN_COMP_GNUC
#define EIGEN_UNUSED __attribute__((unused))
#else
#define EIGEN_UNUSED
#endif

namespace Eigen {
namespace internal {
template<typename T> EIGEN_DEVICE_FUNC void ignore_unused_variable(const T&) {}
}
}
#define EIGEN_UNUSED_VARIABLE(var) Eigen::internal::ignore_unused_variable(var);

#if !defined(EIGEN_ASM_COMMENT)
#if EIGEN_COMP_GNUC && (EIGEN_ARCH_i386_OR_x86_64 || EIGEN_ARCH_ARM_OR_ARM64)
#define EIGEN_ASM_COMMENT(X)  __asm__("#" X)
#else
#define EIGEN_ASM_COMMENT(X)
#endif
#endif


#if EIGEN_COMP_MSVC
#  define EIGEN_CONST_CONDITIONAL(cond)  (void)0, cond
#else
#  define EIGEN_CONST_CONDITIONAL(cond)  cond
#endif

#ifdef EIGEN_DONT_USE_RESTRICT_KEYWORD
#define EIGEN_RESTRICT
#endif
#ifndef EIGEN_RESTRICT
#define EIGEN_RESTRICT __restrict
#endif


#ifndef EIGEN_DEFAULT_IO_FORMAT
#ifdef EIGEN_MAKING_DOCS
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(3, 0, " ", "\n", "", "")
#else
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat()
#endif
#endif

#define EIGEN_EMPTY


#if (defined(EIGEN_CUDA_ARCH) && defined(__NVCC__)) || defined(EIGEN_HIP_DEVICE_COMPILE)
#define EIGEN_USING_STD_MATH(FUNC) using ::FUNC;
#else
#define EIGEN_USING_STD_MATH(FUNC) using std::FUNC;
#endif


#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 || EIGEN_CUDACC_VER>0)
#define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
using Base::operator =;
#elif EIGEN_COMP_CLANG 
#define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
using Base::operator =; \
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) { Base::operator=(other); return *this; } \
template <typename OtherDerived> \
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const DenseBase<OtherDerived>& other) { Base::operator=(other.derived()); return *this; }
#else
#define EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived) \
using Base::operator =; \
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& operator=(const Derived& other) \
{ \
Base::operator=(other); \
return *this; \
}
#endif



#define EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Derived) EIGEN_INHERIT_ASSIGNMENT_EQUAL_OPERATOR(Derived)



#define EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;  \
typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;  \
typedef typename Base::CoeffReturnType CoeffReturnType;  \
typedef typename Eigen::internal::ref_selector<Derived>::type Nested; \
typedef typename Eigen::internal::traits<Derived>::StorageKind StorageKind; \
typedef typename Eigen::internal::traits<Derived>::StorageIndex StorageIndex; \
enum CompileTimeTraits \
{ RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime, \
ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime, \
Flags = Eigen::internal::traits<Derived>::Flags, \
SizeAtCompileTime = Base::SizeAtCompileTime, \
MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime, \
IsVectorAtCompileTime = Base::IsVectorAtCompileTime }; \
using Base::derived; \
using Base::const_cast_derived;


#define EIGEN_DENSE_PUBLIC_INTERFACE(Derived) \
EIGEN_GENERIC_PUBLIC_INTERFACE(Derived) \
typedef typename Base::PacketScalar PacketScalar;


#define EIGEN_PLAIN_ENUM_MIN(a,b) (((int)a <= (int)b) ? (int)a : (int)b)
#define EIGEN_PLAIN_ENUM_MAX(a,b) (((int)a >= (int)b) ? (int)a : (int)b)

#define EIGEN_SIZE_MIN_PREFER_DYNAMIC(a,b) (((int)a == 0 || (int)b == 0) ? 0 \
: ((int)a == 1 || (int)b == 1) ? 1 \
: ((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
: ((int)a <= (int)b) ? (int)a : (int)b)

#define EIGEN_SIZE_MIN_PREFER_FIXED(a,b)  (((int)a == 0 || (int)b == 0) ? 0 \
: ((int)a == 1 || (int)b == 1) ? 1 \
: ((int)a == Dynamic && (int)b == Dynamic) ? Dynamic \
: ((int)a == Dynamic) ? (int)b \
: ((int)b == Dynamic) ? (int)a \
: ((int)a <= (int)b) ? (int)a : (int)b)

#define EIGEN_SIZE_MAX(a,b) (((int)a == Dynamic || (int)b == Dynamic) ? Dynamic \
: ((int)a >= (int)b) ? (int)a : (int)b)

#define EIGEN_LOGICAL_XOR(a,b) (((a) || (b)) && !((a) && (b)))

#define EIGEN_IMPLIES(a,b) (!(a) || (b))

#define EIGEN_CWISE_BINARY_RETURN_TYPE(LHS,RHS,OPNAME) \
CwiseBinaryOp< \
EIGEN_CAT(EIGEN_CAT(internal::scalar_,OPNAME),_op)< \
typename internal::traits<LHS>::Scalar, \
typename internal::traits<RHS>::Scalar \
>, \
const LHS, \
const RHS \
>

#define EIGEN_MAKE_CWISE_BINARY_OP(METHOD,OPNAME) \
template<typename OtherDerived> \
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const EIGEN_CWISE_BINARY_RETURN_TYPE(Derived,OtherDerived,OPNAME) \
(METHOD)(const EIGEN_CURRENT_STORAGE_BASE_CLASS<OtherDerived> &other) const \
{ \
return EIGEN_CWISE_BINARY_RETURN_TYPE(Derived,OtherDerived,OPNAME)(derived(), other.derived()); \
}

#define EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,TYPEA,TYPEB) \
(Eigen::internal::has_ReturnType<Eigen::ScalarBinaryOpTraits<TYPEA,TYPEB,EIGEN_CAT(EIGEN_CAT(Eigen::internal::scalar_,OPNAME),_op)<TYPEA,TYPEB>  > >::value)

#define EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(EXPR,SCALAR,OPNAME) \
CwiseBinaryOp<EIGEN_CAT(EIGEN_CAT(internal::scalar_,OPNAME),_op)<typename internal::traits<EXPR>::Scalar,SCALAR>, const EXPR, \
const typename internal::plain_constant_type<EXPR,SCALAR>::type>

#define EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(SCALAR,EXPR,OPNAME) \
CwiseBinaryOp<EIGEN_CAT(EIGEN_CAT(internal::scalar_,OPNAME),_op)<SCALAR,typename internal::traits<EXPR>::Scalar>, \
const typename internal::plain_constant_type<EXPR,SCALAR>::type, const EXPR>

#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC_STRICT<=1600)
#define EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(X) typename internal::enable_if<true,X>::type
#else
#define EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(X) X
#endif

#define EIGEN_MAKE_SCALAR_BINARY_OP_ONTHERIGHT(METHOD,OPNAME) \
template <typename T> EIGEN_DEVICE_FUNC inline \
EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(Derived,typename internal::promote_scalar_arg<Scalar EIGEN_COMMA T EIGEN_COMMA EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,Scalar,T)>::type,OPNAME))\
(METHOD)(const T& scalar) const { \
typedef typename internal::promote_scalar_arg<Scalar,T,EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,Scalar,T)>::type PromotedT; \
return EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(Derived,PromotedT,OPNAME)(derived(), \
typename internal::plain_constant_type<Derived,PromotedT>::type(derived().rows(), derived().cols(), internal::scalar_constant_op<PromotedT>(scalar))); \
}

#define EIGEN_MAKE_SCALAR_BINARY_OP_ONTHELEFT(METHOD,OPNAME) \
template <typename T> EIGEN_DEVICE_FUNC inline friend \
EIGEN_MSVC10_WORKAROUND_BINARYOP_RETURN_TYPE(const EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(typename internal::promote_scalar_arg<Scalar EIGEN_COMMA T EIGEN_COMMA EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,T,Scalar)>::type,Derived,OPNAME)) \
(METHOD)(const T& scalar, const StorageBaseType& matrix) { \
typedef typename internal::promote_scalar_arg<Scalar,T,EIGEN_SCALAR_BINARY_SUPPORTED(OPNAME,T,Scalar)>::type PromotedT; \
return EIGEN_SCALAR_BINARYOP_EXPR_RETURN_TYPE(PromotedT,Derived,OPNAME)( \
typename internal::plain_constant_type<Derived,PromotedT>::type(matrix.derived().rows(), matrix.derived().cols(), internal::scalar_constant_op<PromotedT>(scalar)), matrix.derived()); \
}

#define EIGEN_MAKE_SCALAR_BINARY_OP(METHOD,OPNAME) \
EIGEN_MAKE_SCALAR_BINARY_OP_ONTHELEFT(METHOD,OPNAME) \
EIGEN_MAKE_SCALAR_BINARY_OP_ONTHERIGHT(METHOD,OPNAME)


#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(EIGEN_CUDA_ARCH) && !defined(EIGEN_EXCEPTIONS) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_HIP_DEVICE_COMPILE)
#define EIGEN_EXCEPTIONS
#endif


#ifdef EIGEN_EXCEPTIONS
#  define EIGEN_THROW_X(X) throw X
#  define EIGEN_THROW throw
#  define EIGEN_TRY try
#  define EIGEN_CATCH(X) catch (X)
#else
#  if defined(EIGEN_CUDA_ARCH)
#    define EIGEN_THROW_X(X) asm("trap;")
#    define EIGEN_THROW asm("trap;")
#  elif defined(EIGEN_HIP_DEVICE_COMPILE)
#    define EIGEN_THROW_X(X) asm("s_trap 0")
#    define EIGEN_THROW asm("s_trap 0")
#  else
#    define EIGEN_THROW_X(X) std::abort()
#    define EIGEN_THROW std::abort()
#  endif
#  define EIGEN_TRY if (true)
#  define EIGEN_CATCH(X) else
#endif


#if EIGEN_HAS_CXX11_NOEXCEPT
#   define EIGEN_INCLUDE_TYPE_TRAITS
#   define EIGEN_NOEXCEPT noexcept
#   define EIGEN_NOEXCEPT_IF(x) noexcept(x)
#   define EIGEN_NO_THROW noexcept(true)
#   define EIGEN_EXCEPTION_SPEC(X) noexcept(false)
#else
#   define EIGEN_NOEXCEPT
#   define EIGEN_NOEXCEPT_IF(x)
#   define EIGEN_NO_THROW throw()
#   if EIGEN_COMP_MSVC
#     define EIGEN_EXCEPTION_SPEC(X) throw()
#   else
#     define EIGEN_EXCEPTION_SPEC(X) throw(X)
#   endif
#endif

#if EIGEN_HAS_VARIADIC_TEMPLATES
namespace Eigen {
namespace internal {

inline bool all(){ return true; }

template<typename T, typename ...Ts>
bool all(T t, Ts ... ts){ return t && all(ts...); }

}
}
#endif

#if EIGEN_HAS_CXX11_OVERRIDE_FINAL
#   define EIGEN_OVERRIDE override
#   define EIGEN_FINAL final
#else
#   define EIGEN_OVERRIDE
#   define EIGEN_FINAL
#endif

#if defined(__SYCL_DEVICE_ONLY__)
#if defined(_MSC_VER)
#define EIGEN_UNROLL_LOOP __pragma(unroll)
#else
#define EIGEN_UNROLL_LOOP _Pragma("unroll")
#endif
#else
#define EIGEN_UNROLL_LOOP
#endif

#endif 
