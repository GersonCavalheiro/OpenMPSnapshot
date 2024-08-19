
#ifndef LLVM_SUPPORT_COMPILER_H
#define LLVM_SUPPORT_COMPILER_H

#include "llvm/Config/llvm-config.h"

#include <new>
#include <stddef.h>

#if defined(_MSC_VER)
#include <sal.h>
#endif

#ifndef __has_feature
# define __has_feature(x) 0
#endif

#ifndef __has_extension
# define __has_extension(x) 0
#endif

#ifndef __has_attribute
# define __has_attribute(x) 0
#endif

#ifndef __has_cpp_attribute
# define __has_cpp_attribute(x) 0
#endif

#ifndef __has_builtin
# define __has_builtin(x) 0
#endif

#ifndef LLVM_GNUC_PREREQ
# if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#  define LLVM_GNUC_PREREQ(maj, min, patch) \
((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >= \
((maj) << 20) + ((min) << 10) + (patch))
# elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#  define LLVM_GNUC_PREREQ(maj, min, patch) \
((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) >= ((maj) << 20) + ((min) << 10))
# else
#  define LLVM_GNUC_PREREQ(maj, min, patch) 0
# endif
#endif

#ifdef _MSC_VER
#define LLVM_MSC_PREREQ(version) (_MSC_VER >= (version))

#if !LLVM_MSC_PREREQ(1900)
#error LLVM requires at least MSVC 2015.
#endif

#else
#define LLVM_MSC_PREREQ(version) 0
#endif

#if __has_feature(cxx_rvalue_references) || LLVM_GNUC_PREREQ(4, 8, 1)
#define LLVM_HAS_RVALUE_REFERENCE_THIS 1
#else
#define LLVM_HAS_RVALUE_REFERENCE_THIS 0
#endif

#if LLVM_HAS_RVALUE_REFERENCE_THIS
#define LLVM_LVALUE_FUNCTION &
#else
#define LLVM_LVALUE_FUNCTION
#endif

#if (__has_attribute(visibility) || LLVM_GNUC_PREREQ(4, 0, 0)) &&              \
!defined(__MINGW32__) && !defined(__CYGWIN__) && !defined(_WIN32)
#define LLVM_LIBRARY_VISIBILITY __attribute__ ((visibility("hidden")))
#else
#define LLVM_LIBRARY_VISIBILITY
#endif

#if defined(__GNUC__)
#define LLVM_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#else
#define LLVM_PREFETCH(addr, rw, locality)
#endif

#if __has_attribute(used) || LLVM_GNUC_PREREQ(3, 1, 0)
#define LLVM_ATTRIBUTE_USED __attribute__((__used__))
#else
#define LLVM_ATTRIBUTE_USED
#endif

#if __cplusplus > 201402L && __has_cpp_attribute(nodiscard)
#define LLVM_NODISCARD [[nodiscard]]
#elif !__cplusplus
#define LLVM_NODISCARD
#elif __has_cpp_attribute(clang::warn_unused_result)
#define LLVM_NODISCARD [[clang::warn_unused_result]]
#else
#define LLVM_NODISCARD
#endif

#if __has_attribute(unused) || LLVM_GNUC_PREREQ(3, 1, 0)
#define LLVM_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define LLVM_ATTRIBUTE_UNUSED
#endif

#if (__has_attribute(weak) || LLVM_GNUC_PREREQ(4, 0, 0)) &&                    \
(!defined(__MINGW32__) && !defined(__CYGWIN__) && !defined(_WIN32))
#define LLVM_ATTRIBUTE_WEAK __attribute__((__weak__))
#else
#define LLVM_ATTRIBUTE_WEAK
#endif

#if defined(__clang__) || defined(__GNUC__)
#define LLVM_READNONE __attribute__((__const__))
#else
#define LLVM_READNONE
#endif

#if __has_attribute(pure) || defined(__GNUC__)
#define LLVM_READONLY __attribute__((__pure__))
#else
#define LLVM_READONLY
#endif

#if __has_builtin(__builtin_expect) || LLVM_GNUC_PREREQ(4, 0, 0)
#define LLVM_LIKELY(EXPR) __builtin_expect((bool)(EXPR), true)
#define LLVM_UNLIKELY(EXPR) __builtin_expect((bool)(EXPR), false)
#else
#define LLVM_LIKELY(EXPR) (EXPR)
#define LLVM_UNLIKELY(EXPR) (EXPR)
#endif

#if __has_attribute(noinline) || LLVM_GNUC_PREREQ(3, 4, 0)
#define LLVM_ATTRIBUTE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define LLVM_ATTRIBUTE_NOINLINE
#endif

#if __has_attribute(always_inline) || LLVM_GNUC_PREREQ(4, 0, 0)
#define LLVM_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_ALWAYS_INLINE __forceinline
#else
#define LLVM_ATTRIBUTE_ALWAYS_INLINE
#endif

#ifdef __GNUC__
#define LLVM_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define LLVM_ATTRIBUTE_NORETURN
#endif

#if __has_attribute(returns_nonnull) || LLVM_GNUC_PREREQ(4, 9, 0)
#define LLVM_ATTRIBUTE_RETURNS_NONNULL __attribute__((returns_nonnull))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_RETURNS_NONNULL _Ret_notnull_
#else
#define LLVM_ATTRIBUTE_RETURNS_NONNULL
#endif

#ifdef __GNUC__
#define LLVM_ATTRIBUTE_RETURNS_NOALIAS __attribute__((__malloc__))
#elif defined(_MSC_VER)
#define LLVM_ATTRIBUTE_RETURNS_NOALIAS __declspec(restrict)
#else
#define LLVM_ATTRIBUTE_RETURNS_NOALIAS
#endif

#if __cplusplus > 201402L && __has_cpp_attribute(fallthrough)
#define LLVM_FALLTHROUGH [[fallthrough]]
#elif __has_cpp_attribute(gnu::fallthrough)
#define LLVM_FALLTHROUGH [[gnu::fallthrough]]
#elif !__cplusplus
#define LLVM_FALLTHROUGH
#elif __has_cpp_attribute(clang::fallthrough)
#define LLVM_FALLTHROUGH [[clang::fallthrough]]
#else
#define LLVM_FALLTHROUGH
#endif

#ifdef __GNUC__
#define LLVM_EXTENSION __extension__
#else
#define LLVM_EXTENSION
#endif

#if __has_feature(attribute_deprecated_with_message)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
decl __attribute__((deprecated(message)))
#elif defined(__GNUC__)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
decl __attribute__((deprecated))
#elif defined(_MSC_VER)
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
__declspec(deprecated(message)) decl
#else
# define LLVM_ATTRIBUTE_DEPRECATED(decl, message) \
decl
#endif

#if __has_builtin(__builtin_unreachable) || LLVM_GNUC_PREREQ(4, 5, 0)
# define LLVM_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
# define LLVM_BUILTIN_UNREACHABLE __assume(false)
#endif

#if __has_builtin(__builtin_trap) || LLVM_GNUC_PREREQ(4, 3, 0)
# define LLVM_BUILTIN_TRAP __builtin_trap()
#elif defined(_MSC_VER)
# define LLVM_BUILTIN_TRAP __debugbreak()
#else
# define LLVM_BUILTIN_TRAP *(volatile int*)0x11 = 0
#endif

#if __has_builtin(__builtin_debugtrap)
# define LLVM_BUILTIN_DEBUGTRAP __builtin_debugtrap()
#elif defined(_MSC_VER)
# define LLVM_BUILTIN_DEBUGTRAP __debugbreak()
#else
# define LLVM_BUILTIN_DEBUGTRAP
#endif

#if __has_builtin(__builtin_assume_aligned) || LLVM_GNUC_PREREQ(4, 7, 0)
# define LLVM_ASSUME_ALIGNED(p, a) __builtin_assume_aligned(p, a)
#elif defined(LLVM_BUILTIN_UNREACHABLE)
# define LLVM_ASSUME_ALIGNED(p, a) \
(((uintptr_t(p) % (a)) == 0) ? (p) : (LLVM_BUILTIN_UNREACHABLE, (p)))
#else
# define LLVM_ASSUME_ALIGNED(p, a) (p)
#endif

#if __GNUC__ && !__has_feature(cxx_alignas) && !LLVM_GNUC_PREREQ(4, 8, 1)
# define LLVM_ALIGNAS(x) __attribute__((aligned(x)))
#else
# define LLVM_ALIGNAS(x) alignas(x)
#endif

#ifdef _MSC_VER
# define LLVM_PACKED(d) __pragma(pack(push, 1)) d __pragma(pack(pop))
# define LLVM_PACKED_START __pragma(pack(push, 1))
# define LLVM_PACKED_END   __pragma(pack(pop))
#else
# define LLVM_PACKED(d) d __attribute__((packed))
# define LLVM_PACKED_START _Pragma("pack(push, 1)")
# define LLVM_PACKED_END   _Pragma("pack(pop)")
#endif

#ifdef __SIZEOF_POINTER__
# define LLVM_PTR_SIZE __SIZEOF_POINTER__
#elif defined(_WIN64)
# define LLVM_PTR_SIZE 8
#elif defined(_WIN32)
# define LLVM_PTR_SIZE 4
#elif defined(_MSC_VER)
# error "could not determine LLVM_PTR_SIZE as a constant int for MSVC"
#else
# define LLVM_PTR_SIZE sizeof(void *)
#endif

#if __has_feature(memory_sanitizer)
# define LLVM_MEMORY_SANITIZER_BUILD 1
# include <sanitizer/msan_interface.h>
#else
# define LLVM_MEMORY_SANITIZER_BUILD 0
# define __msan_allocated_memory(p, size)
# define __msan_unpoison(p, size)
#endif

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
# define LLVM_ADDRESS_SANITIZER_BUILD 1
# include <sanitizer/asan_interface.h>
#else
# define LLVM_ADDRESS_SANITIZER_BUILD 0
# define __asan_poison_memory_region(p, size)
# define __asan_unpoison_memory_region(p, size)
#endif

#if __has_feature(thread_sanitizer) || defined(__SANITIZE_THREAD__)
# define LLVM_THREAD_SANITIZER_BUILD 1
#else
# define LLVM_THREAD_SANITIZER_BUILD 0
#endif

#if LLVM_THREAD_SANITIZER_BUILD
#ifdef __cplusplus
extern "C" {
#endif
void AnnotateHappensAfter(const char *file, int line, const volatile void *cv);
void AnnotateHappensBefore(const char *file, int line, const volatile void *cv);
void AnnotateIgnoreWritesBegin(const char *file, int line);
void AnnotateIgnoreWritesEnd(const char *file, int line);
#ifdef __cplusplus
}
#endif

# define TsanHappensBefore(cv) AnnotateHappensBefore(__FILE__, __LINE__, cv)

# define TsanHappensAfter(cv) AnnotateHappensAfter(__FILE__, __LINE__, cv)

# define TsanIgnoreWritesBegin() AnnotateIgnoreWritesBegin(__FILE__, __LINE__)

# define TsanIgnoreWritesEnd() AnnotateIgnoreWritesEnd(__FILE__, __LINE__)
#else
# define TsanHappensBefore(cv)
# define TsanHappensAfter(cv)
# define TsanIgnoreWritesBegin()
# define TsanIgnoreWritesEnd()
#endif

#if __has_attribute(no_sanitize)
#define LLVM_NO_SANITIZE(KIND) __attribute__((no_sanitize(KIND)))
#else
#define LLVM_NO_SANITIZE(KIND)
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
#define LLVM_DUMP_METHOD LLVM_ATTRIBUTE_NOINLINE LLVM_ATTRIBUTE_USED
#else
#define LLVM_DUMP_METHOD LLVM_ATTRIBUTE_NOINLINE
#endif

#if defined(_MSC_VER)
#define LLVM_PRETTY_FUNCTION __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
#define LLVM_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define LLVM_PRETTY_FUNCTION __func__
#endif

#if LLVM_ENABLE_THREADS
#if __has_feature(cxx_thread_local)
#define LLVM_THREAD_LOCAL thread_local
#elif defined(_MSC_VER)
#define LLVM_THREAD_LOCAL __declspec(thread)
#else
#define LLVM_THREAD_LOCAL __thread
#endif
#else 
#define LLVM_THREAD_LOCAL
#endif

#if __has_feature(cxx_exceptions)
#define LLVM_ENABLE_EXCEPTIONS 1
#elif defined(__GNUC__) && defined(__EXCEPTIONS)
#define LLVM_ENABLE_EXCEPTIONS 1
#elif defined(_MSC_VER) && defined(_CPPUNWIND)
#define LLVM_ENABLE_EXCEPTIONS 1
#endif

namespace llvm {

inline void *allocate_buffer(size_t Size, size_t Alignment) {
return ::operator new(Size
#if __cpp_aligned_new
,
std::align_val_t(Alignment)
#endif
);
}

inline void deallocate_buffer(void *Ptr, size_t Size, size_t Alignment) {
::operator delete(Ptr
#if __cpp_sized_deallocation
,
Size
#endif
#if __cpp_aligned_new
,
std::align_val_t(Alignment)
#endif
);
}

} 

#endif
