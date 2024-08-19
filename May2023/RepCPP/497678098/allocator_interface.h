

#ifndef SEQAN_INCLUDE_SEQAN_BASIC_ALLOCATOR_INTERFACE_H_
#define SEQAN_INCLUDE_SEQAN_BASIC_ALLOCATOR_INTERFACE_H_

namespace seqan {


struct Tristate_;
typedef Tag<Tristate_> Tristate;
template <typename TValue, typename TSpec> struct Holder;





struct AllocateUnspecified_;
typedef Tag<AllocateUnspecified_> TagAllocateUnspecified;

struct AllocateTemp_;
typedef Tag<AllocateTemp_> TagAllocateTemp;

struct AllocateStorage_;
typedef Tag<AllocateStorage_> TagAllocateStorage;

struct AllocateAlignedMalloc_;
typedef Tag<AllocateAlignedMalloc_> TagAllocateAlignedMalloc;



template <typename TSpec>
struct Allocator;



template <typename TSpec>
struct Spec<Allocator<TSpec> >
{
typedef TSpec Type;
};





template <typename T, typename TValue, typename TSize>
inline void
allocate(T const & me,
TValue * & data,
TSize count)
{
allocate(me, data, count, TagAllocateUnspecified());
}

template <typename T, typename TValue, typename TSize>
inline void
allocate(T & me,
TValue * & data,
TSize count)
{
allocate(me, data, count, TagAllocateUnspecified());
}

template <typename T, typename TValue, typename TSize, typename TUsage>
inline void
allocate(T const &,
TValue * & data,
TSize count,
Tag<TUsage> const &)
{
#ifdef STDLIB_VS
data = (TValue *) _aligned_malloc(count * sizeof(TValue), __alignof(TValue));
#else

SEQAN_ASSERT_LEQ(static_cast<std::size_t>(count), std::numeric_limits<std::size_t>::max() / sizeof(TValue));
#   if defined(COMPILER_GCC) &&  __GNUC__ >= 7
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Walloc-size-larger-than="
#   endif 
data = (TValue *) operator new(count * sizeof(TValue));
#   if defined(COMPILER_GCC) &&  __GNUC__ >= 7
#       pragma GCC diagnostic pop
#   endif 
#endif

#ifdef SEQAN_PROFILE
if (data)
SEQAN_PROADD(SEQAN_PROMEMORY, count * sizeof(TValue));
#endif
}

template <typename T, typename TValue, typename TSize>
inline void
allocate(T const &,
TValue * & data,
TSize count,
TagAllocateAlignedMalloc const &)
{
#ifdef PLATFORM_WINDOWS_VS
data = (TValue *) _aligned_malloc(count * sizeof(TValue), __alignof(TValue));
#else
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
const size_t align = (__alignof__(TValue) < sizeof(void*)) ? sizeof(void*): __alignof__(TValue);
if (posix_memalign(&(void* &)data, align, count * sizeof(TValue)))
data = NULL;
#else
data = (TValue *) malloc(count * sizeof(TValue));
#endif
#endif
}

template <typename T, typename TValue, typename TSize>
inline void
allocate(T &,
TValue * & data,
TSize count,
TagAllocateAlignedMalloc const &)
{
#ifdef PLATFORM_WINDOWS_VS
data = (TValue *) _aligned_malloc(count * sizeof(TValue), __alignof(TValue));
#else
#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
const size_t align = (__alignof__(TValue) < sizeof(void*)) ? sizeof(void*) : __alignof__(TValue);
if (posix_memalign(&(void* &)data, align, count * sizeof(TValue)))
data = NULL;
#else
data = (TValue *) malloc(count * sizeof(TValue));
#endif
#endif
}




template <typename T, typename TValue, typename TSize>
inline void
deallocate(T const & me,
TValue * data,
TSize const count)
{
deallocate(me, data, count, TagAllocateUnspecified());
}

template <typename T, typename TValue, typename TSize>
inline void
deallocate(T & me,
TValue * data,
TSize const count)
{
deallocate(me, data, count, TagAllocateUnspecified());
}

template <typename T, typename TValue, typename TSize, typename TUsage>
inline void
deallocate(
T const & ,
TValue * data,
#ifdef SEQAN_PROFILE
TSize count,
#else
TSize,
#endif
Tag<TUsage> const)
{
#ifdef SEQAN_PROFILE
if (data && count)  
SEQAN_PROSUB(SEQAN_PROMEMORY, count * sizeof(TValue));
#endif
#ifdef STDLIB_VS
_aligned_free((void *) data);
#else
operator delete ((void *) data);
#endif
}

template <typename T, typename TValue, typename TSize, typename TUsage>
inline void
deallocate(
T & ,
TValue * data,
#ifdef SEQAN_PROFILE
TSize count,
#else
TSize,
#endif
Tag<TUsage> const)
{
#ifdef SEQAN_PROFILE
if (data && count)  
SEQAN_PROSUB(SEQAN_PROMEMORY, count * sizeof(TValue));
#endif
#ifdef STDLIB_VS
_aligned_free((void *) data);
#else
operator delete ((void *) data);
#endif
}

template <typename T, typename TValue, typename TSize>
inline void
deallocate(
T const & ,
TValue * data,
#ifdef SEQAN_PROFILE
TSize count,
#else
TSize,
#endif
TagAllocateAlignedMalloc const)
{
#ifdef SEQAN_PROFILE
if (data && count)  
SEQAN_PROSUB(SEQAN_PROMEMORY, count * sizeof(TValue));
#endif
#ifdef PLATFORM_WINDOWS_VS
_aligned_free((void *) data);
#else
free((void *) data);
#endif
}

template <typename T, typename TValue, typename TSize>
inline void
deallocate(
T & ,
TValue * data,
#ifdef SEQAN_PROFILE
TSize count,
#else
TSize,
#endif
TagAllocateAlignedMalloc const)
{
#ifdef SEQAN_PROFILE
if (data && count)  
SEQAN_PROSUB(SEQAN_PROMEMORY, count * sizeof(TValue));
#endif
#ifdef PLATFORM_WINDOWS_VS
_aligned_free((void *) data);
#else
free((void *) data);
#endif
}

}  

#endif  
