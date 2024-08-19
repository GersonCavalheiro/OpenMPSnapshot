


#ifndef ABSL_BASE_DYNAMIC_ANNOTATIONS_H_
#define ABSL_BASE_DYNAMIC_ANNOTATIONS_H_

#ifndef DYNAMIC_ANNOTATIONS_ENABLED
# define DYNAMIC_ANNOTATIONS_ENABLED 0
#endif

#if DYNAMIC_ANNOTATIONS_ENABLED != 0




#define ANNOTATE_BENIGN_RACE(pointer, description) \
AnnotateBenignRaceSized(__FILE__, __LINE__, pointer, \
sizeof(*(pointer)), description)


#define ANNOTATE_BENIGN_RACE_SIZED(address, size, description) \
AnnotateBenignRaceSized(__FILE__, __LINE__, address, size, description)


#define ANNOTATE_ENABLE_RACE_DETECTION(enable) \
AnnotateEnableRaceDetection(__FILE__, __LINE__, enable)




#define ANNOTATE_THREAD_NAME(name) \
AnnotateThreadName(__FILE__, __LINE__, name)




#define ANNOTATE_RWLOCK_CREATE(lock) \
AnnotateRWLockCreate(__FILE__, __LINE__, lock)


#ifdef THREAD_SANITIZER
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock) \
AnnotateRWLockCreateStatic(__FILE__, __LINE__, lock)
#else
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock) ANNOTATE_RWLOCK_CREATE(lock)
#endif


#define ANNOTATE_RWLOCK_DESTROY(lock) \
AnnotateRWLockDestroy(__FILE__, __LINE__, lock)


#define ANNOTATE_RWLOCK_ACQUIRED(lock, is_w) \
AnnotateRWLockAcquired(__FILE__, __LINE__, lock, is_w)


#define ANNOTATE_RWLOCK_RELEASED(lock, is_w) \
AnnotateRWLockReleased(__FILE__, __LINE__, lock, is_w)

#else  

#define ANNOTATE_RWLOCK_CREATE(lock) 
#define ANNOTATE_RWLOCK_CREATE_STATIC(lock) 
#define ANNOTATE_RWLOCK_DESTROY(lock) 
#define ANNOTATE_RWLOCK_ACQUIRED(lock, is_w) 
#define ANNOTATE_RWLOCK_RELEASED(lock, is_w) 
#define ANNOTATE_BENIGN_RACE(address, description) 
#define ANNOTATE_BENIGN_RACE_SIZED(address, size, description) 
#define ANNOTATE_THREAD_NAME(name) 
#define ANNOTATE_ENABLE_RACE_DETECTION(enable) 

#endif  


#if DYNAMIC_ANNOTATIONS_ENABLED == 1 || defined(MEMORY_SANITIZER)
#define ANNOTATE_MEMORY_IS_INITIALIZED(address, size) \
AnnotateMemoryIsInitialized(__FILE__, __LINE__, address, size)

#define ANNOTATE_MEMORY_IS_UNINITIALIZED(address, size) \
AnnotateMemoryIsUninitialized(__FILE__, __LINE__, address, size)
#else
#define ANNOTATE_MEMORY_IS_INITIALIZED(address, size) 
#define ANNOTATE_MEMORY_IS_UNINITIALIZED(address, size) 
#endif  

#if defined(__clang__) && (!defined(SWIG)) \
&& defined(__CLANG_SUPPORT_DYN_ANNOTATION__)

#if DYNAMIC_ANNOTATIONS_ENABLED == 0
#define ANNOTALYSIS_ENABLED
#endif


#define ATTRIBUTE_IGNORE_READS_BEGIN \
__attribute((exclusive_lock_function("*")))
#define ATTRIBUTE_IGNORE_READS_END \
__attribute((unlock_function("*")))
#else
#define ATTRIBUTE_IGNORE_READS_BEGIN  
#define ATTRIBUTE_IGNORE_READS_END  
#endif  

#if (DYNAMIC_ANNOTATIONS_ENABLED != 0) || defined(ANNOTALYSIS_ENABLED)
#define ANNOTATIONS_ENABLED
#endif

#if (DYNAMIC_ANNOTATIONS_ENABLED != 0)


#define ANNOTATE_IGNORE_READS_BEGIN() \
AnnotateIgnoreReadsBegin(__FILE__, __LINE__)


#define ANNOTATE_IGNORE_READS_END() \
AnnotateIgnoreReadsEnd(__FILE__, __LINE__)


#define ANNOTATE_IGNORE_WRITES_BEGIN() \
AnnotateIgnoreWritesBegin(__FILE__, __LINE__)


#define ANNOTATE_IGNORE_WRITES_END() \
AnnotateIgnoreWritesEnd(__FILE__, __LINE__)


#elif defined(ANNOTALYSIS_ENABLED)

#define ANNOTATE_IGNORE_READS_BEGIN() \
StaticAnnotateIgnoreReadsBegin(__FILE__, __LINE__)

#define ANNOTATE_IGNORE_READS_END() \
StaticAnnotateIgnoreReadsEnd(__FILE__, __LINE__)

#define ANNOTATE_IGNORE_WRITES_BEGIN() \
StaticAnnotateIgnoreWritesBegin(__FILE__, __LINE__)

#define ANNOTATE_IGNORE_WRITES_END() \
StaticAnnotateIgnoreWritesEnd(__FILE__, __LINE__)

#else
#define ANNOTATE_IGNORE_READS_BEGIN()  
#define ANNOTATE_IGNORE_READS_END()  
#define ANNOTATE_IGNORE_WRITES_BEGIN()  
#define ANNOTATE_IGNORE_WRITES_END()  
#endif


#if defined(ANNOTATIONS_ENABLED)


#define ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN() \
do {                                           \
ANNOTATE_IGNORE_READS_BEGIN();               \
ANNOTATE_IGNORE_WRITES_BEGIN();              \
}while (0)


#define ANNOTATE_IGNORE_READS_AND_WRITES_END()   \
do {                                           \
ANNOTATE_IGNORE_WRITES_END();                \
ANNOTATE_IGNORE_READS_END();                 \
}while (0)

#else
#define ANNOTATE_IGNORE_READS_AND_WRITES_BEGIN()  
#define ANNOTATE_IGNORE_READS_AND_WRITES_END()  
#endif


#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void AnnotateRWLockCreate(const char *file, int line,
const volatile void *lock);
void AnnotateRWLockCreateStatic(const char *file, int line,
const volatile void *lock);
void AnnotateRWLockDestroy(const char *file, int line,
const volatile void *lock);
void AnnotateRWLockAcquired(const char *file, int line,
const volatile void *lock, long is_w);  
void AnnotateRWLockReleased(const char *file, int line,
const volatile void *lock, long is_w);  
void AnnotateBenignRace(const char *file, int line,
const volatile void *address,
const char *description);
void AnnotateBenignRaceSized(const char *file, int line,
const volatile void *address,
size_t size,
const char *description);
void AnnotateThreadName(const char *file, int line,
const char *name);
void AnnotateEnableRaceDetection(const char *file, int line, int enable);
void AnnotateMemoryIsInitialized(const char *file, int line,
const volatile void *mem, size_t size);
void AnnotateMemoryIsUninitialized(const char *file, int line,
const volatile void *mem, size_t size);


void AnnotateIgnoreReadsBegin(const char *file, int line)
ATTRIBUTE_IGNORE_READS_BEGIN;
void AnnotateIgnoreReadsEnd(const char *file, int line)
ATTRIBUTE_IGNORE_READS_END;
void AnnotateIgnoreWritesBegin(const char *file, int line);
void AnnotateIgnoreWritesEnd(const char *file, int line);

#if defined(ANNOTALYSIS_ENABLED)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static inline void StaticAnnotateIgnoreReadsBegin(const char *file, int line)
ATTRIBUTE_IGNORE_READS_BEGIN { (void)file; (void)line; }
static inline void StaticAnnotateIgnoreReadsEnd(const char *file, int line)
ATTRIBUTE_IGNORE_READS_END { (void)file; (void)line; }
static inline void StaticAnnotateIgnoreWritesBegin(
const char *file, int line) { (void)file; (void)line; }
static inline void StaticAnnotateIgnoreWritesEnd(
const char *file, int line) { (void)file; (void)line; }
#pragma GCC diagnostic pop
#endif


int RunningOnValgrind(void);


double ValgrindSlowdown(void);

#ifdef __cplusplus
}
#endif


#if defined(__cplusplus) && defined(ANNOTATIONS_ENABLED)
template <typename T>
inline T ANNOTATE_UNPROTECTED_READ(const volatile T &x) { 
ANNOTATE_IGNORE_READS_BEGIN();
T res = x;
ANNOTATE_IGNORE_READS_END();
return res;
}
#else
#define ANNOTATE_UNPROTECTED_READ(x) (x)
#endif

#if DYNAMIC_ANNOTATIONS_ENABLED != 0 && defined(__cplusplus)

#define ANNOTATE_BENIGN_RACE_STATIC(static_var, description)        \
namespace {                                                       \
class static_var ## _annotator {                                \
public:                                                        \
static_var ## _annotator() {                                  \
ANNOTATE_BENIGN_RACE_SIZED(&static_var,                     \
sizeof(static_var),             \
# static_var ": " description);                           \
}                                                             \
};                                                              \
static static_var ## _annotator the ## static_var ## _annotator;\
}  
#else 
#define ANNOTATE_BENIGN_RACE_STATIC(static_var, description)  
#endif 

#ifdef ADDRESS_SANITIZER

#include <sanitizer/common_interface_defs.h>
#define ANNOTATE_CONTIGUOUS_CONTAINER(beg, end, old_mid, new_mid) \
__sanitizer_annotate_contiguous_container(beg, end, old_mid, new_mid)
#define ADDRESS_SANITIZER_REDZONE(name)         \
struct { char x[8] __attribute__ ((aligned (8))); } name
#else
#define ANNOTATE_CONTIGUOUS_CONTAINER(beg, end, old_mid, new_mid)
#define ADDRESS_SANITIZER_REDZONE(name)
#endif  


#undef ANNOTALYSIS_ENABLED
#undef ANNOTATIONS_ENABLED
#undef ATTRIBUTE_IGNORE_READS_BEGIN
#undef ATTRIBUTE_IGNORE_READS_END

#endif  
