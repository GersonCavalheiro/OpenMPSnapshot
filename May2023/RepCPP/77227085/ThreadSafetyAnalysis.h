#pragma once


#ifndef THREAD_SAFETY_ANALYSIS_MUTEX_H
#define THREAD_SAFETY_ANALYSIS_MUTEX_H

#pragma GCC system_header

#if defined(__clang__) && (!defined(SWIG))
#define THREAD_ANNOTATION_ATTRIBUTE__(x) __attribute__((x))
#else
#define THREAD_ANNOTATION_ATTRIBUTE__(x) 
#endif

#define CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(capability(x))

#define SCOPED_CAPABILITY THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(guarded_by(x))

#define PT_GUARDED_BY(x) THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_by(x))

#define ACQUIRED_BEFORE(...)                                                   \
THREAD_ANNOTATION_ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define ACQUIRED_AFTER(...)                                                    \
THREAD_ANNOTATION_ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define REQUIRES(...)                                                          \
THREAD_ANNOTATION_ATTRIBUTE__(requires_capability(__VA_ARGS__))

#define REQUIRES_SHARED(...)                                                   \
THREAD_ANNOTATION_ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

#define ACQUIRE(...)                                                           \
THREAD_ANNOTATION_ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define ACQUIRE_SHARED(...)                                                    \
THREAD_ANNOTATION_ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define RELEASE(...)                                                           \
THREAD_ANNOTATION_ATTRIBUTE__(release_capability(__VA_ARGS__))

#define RELEASE_SHARED(...)                                                    \
THREAD_ANNOTATION_ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

#define TRY_ACQUIRE(...)                                                       \
THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

#define TRY_ACQUIRE_SHARED(...)                                                \
THREAD_ANNOTATION_ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

#define EXCLUDES(...) THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define ASSERT_CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(assert_capability(x))

#define ASSERT_SHARED_CAPABILITY(x)                                            \
THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_capability(x))

#define RETURN_CAPABILITY(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#define NO_THREAD_SAFETY_ANALYSIS                                              \
THREAD_ANNOTATION_ATTRIBUTE__(no_thread_safety_analysis)

#ifdef USE_LOCK_STYLE_THREAD_SAFETY_ATTRIBUTES

#define PT_GUARDED_VAR THREAD_ANNOTATION_ATTRIBUTE__(pt_guarded_var)

#define GUARDED_VAR THREAD_ANNOTATION_ATTRIBUTE__(guarded_var)

#define EXCLUSIVE_LOCKS_REQUIRED(...)                                          \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_locks_required(__VA_ARGS__))

#define SHARED_LOCKS_REQUIRED(...)                                             \
THREAD_ANNOTATION_ATTRIBUTE__(shared_locks_required(__VA_ARGS__))

#define LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(lockable)

#define SCOPED_LOCKABLE THREAD_ANNOTATION_ATTRIBUTE__(scoped_lockable)

#define EXCLUSIVE_LOCK_FUNCTION(...)                                           \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_lock_function(__VA_ARGS__))

#define SHARED_LOCK_FUNCTION(...)                                              \
THREAD_ANNOTATION_ATTRIBUTE__(shared_lock_function(__VA_ARGS__))

#define UNLOCK_FUNCTION(...)                                                   \
THREAD_ANNOTATION_ATTRIBUTE__(unlock_function(__VA_ARGS__))

#define EXCLUSIVE_TRYLOCK_FUNCTION(...)                                        \
THREAD_ANNOTATION_ATTRIBUTE__(exclusive_trylock_function(__VA_ARGS__))

#define SHARED_TRYLOCK_FUNCTION(...)                                           \
THREAD_ANNOTATION_ATTRIBUTE__(shared_trylock_function(__VA_ARGS__))

#define ASSERT_EXCLUSIVE_LOCK(...)                                             \
THREAD_ANNOTATION_ATTRIBUTE__(assert_exclusive_lock(__VA_ARGS__))

#define ASSERT_SHARED_LOCK(...)                                                \
THREAD_ANNOTATION_ATTRIBUTE__(assert_shared_lock(__VA_ARGS__))

#define LOCKS_EXCLUDED(...)                                                    \
THREAD_ANNOTATION_ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define LOCK_RETURNED(x) THREAD_ANNOTATION_ATTRIBUTE__(lock_returned(x))

#endif 

#endif 
