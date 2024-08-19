


#ifndef TENSORFLOW_CORE_PLATFORM_DEFAULT_THREAD_ANNOTATIONS_H_
#define TENSORFLOW_CORE_PLATFORM_DEFAULT_THREAD_ANNOTATIONS_H_


#if defined(__clang__) && (!defined(SWIG))
#define THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define THREAD_ANNOTATION_ATTRIBUTE__(x)  
#endif

#define GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))
#define GUARDED_VAR  

#define PT_GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))
#define PT_GUARDED_VAR  

#define ACQUIRED_AFTER(...) \
THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define ACQUIRED_BEFORE(...) \
THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define ACQUIRE(...) \
THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define ACQUIRE_SHARED(...) \
THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define RELEASE(...) \
THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

#define EXCLUSIVE_LOCKS_REQUIRED(...) \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_locks_required(__VA_ARGS__))

#define SHARED_LOCKS_REQUIRED(...) \
THREAD_ANNOTATION_ATTRIBUTE__(shared_locks_required(__VA_ARGS__))

#define LOCKS_EXCLUDED(...) \
THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define LOCK_RETURNED(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#define LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(lockable)

#define SCOPED_LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define EXCLUSIVE_LOCK_FUNCTION(...) \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_lock_function(__VA_ARGS__))

#define SHARED_LOCK_FUNCTION(...) \
THREAD_ANNOTATION_ATTRIBUTE__(shared_lock_function(__VA_ARGS__))

#define UNLOCK_FUNCTION(...) \
THREAD_ANNOTATION_ATTRIBUTE__(unlock_function(__VA_ARGS__))

#define EXCLUSIVE_TRYLOCK_FUNCTION(...) \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_trylock_function(__VA_ARGS__))

#define SHARED_TRYLOCK_FUNCTION(...) \
THREAD_ANNOTATION_ATTRIBUTE__(shared_trylock_function(__VA_ARGS__))

#define ASSERT_EXCLUSIVE_LOCK(...) \
THREAD_ANNOTATION_ATTRIBUTE__(assert_exclusive_lock(__VA_ARGS__))

#define ASSERT_SHARED_LOCK(...) \
THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_lock(__VA_ARGS__))

#define NO_THREAD_SAFETY_ANALYSIS \
THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

#define TS_UNCHECKED(x) ""

namespace tensorflow {
namespace thread_safety_analysis {

template <class T>
inline const T& ts_unchecked_read(const T& v) NO_THREAD_SAFETY_ANALYSIS {
return v;
}

template <class T>
inline T& ts_unchecked_read(T& v) NO_THREAD_SAFETY_ANALYSIS {
return v;
}
}  
}  

#endif  
