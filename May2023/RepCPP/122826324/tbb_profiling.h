

#ifndef __TBB_profiling_H
#define __TBB_profiling_H

#if (_WIN32||_WIN64||__linux__) && !__MINGW32__ && TBB_USE_THREADING_TOOLS

#if _WIN32||_WIN64
#include <stdlib.h>  
#endif
#include "tbb_stddef.h"

namespace tbb {
namespace internal {
#if _WIN32||_WIN64
void __TBB_EXPORTED_FUNC itt_set_sync_name_v3( void *obj, const wchar_t* name );
inline size_t multibyte_to_widechar( wchar_t* wcs, const char* mbs, size_t bufsize) {
#if _MSC_VER>=1400
size_t len;
mbstowcs_s( &len, wcs, bufsize, mbs, _TRUNCATE );
return len;   
#else
size_t len = mbstowcs( wcs, mbs, bufsize );
if(wcs && len!=size_t(-1) )
wcs[len<bufsize-1? len: bufsize-1] = wchar_t('\0');
return len+1; 
#endif
}
#else
void __TBB_EXPORTED_FUNC itt_set_sync_name_v3( void *obj, const char* name );
#endif
} 
} 


#if _WIN32||_WIN64
#define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)    \
namespace profiling {                                                       \
inline void set_name( sync_object_type& obj, const wchar_t* name ) {    \
tbb::internal::itt_set_sync_name_v3( &obj, name );                  \
}                                                                       \
inline void set_name( sync_object_type& obj, const char* name ) {       \
size_t len = tbb::internal::multibyte_to_widechar(NULL, name, 0);   \
wchar_t *wname = new wchar_t[len];                                  \
tbb::internal::multibyte_to_widechar(wname, name, len);             \
set_name( obj, wname );                                             \
delete[] wname;                                                     \
}                                                                       \
}
#else 
#define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)    \
namespace profiling {                                                       \
inline void set_name( sync_object_type& obj, const char* name ) {       \
tbb::internal::itt_set_sync_name_v3( &obj, name );                  \
}                                                                       \
}
#endif 

#else 

#if _WIN32||_WIN64
#define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)    \
namespace profiling {                                               \
inline void set_name( sync_object_type&, const wchar_t* ) {}    \
inline void set_name( sync_object_type&, const char* ) {}       \
}
#else 
#define __TBB_DEFINE_PROFILING_SET_NAME(sync_object_type)    \
namespace profiling {                                               \
inline void set_name( sync_object_type&, const char* ) {}       \
}
#endif 

#endif 

#include "atomic.h"
namespace tbb {
namespace internal {

enum notify_type {prepare=0, cancel, acquired, releasing};
const uintptr_t NUM_NOTIFY_TYPES = 4; 

void __TBB_EXPORTED_FUNC call_itt_notify_v5(int t, void *ptr);
void __TBB_EXPORTED_FUNC itt_store_pointer_with_release_v3(void *dst, void *src);
void* __TBB_EXPORTED_FUNC itt_load_pointer_with_acquire_v3(const void *src);
void* __TBB_EXPORTED_FUNC itt_load_pointer_v3( const void* src );

template <typename T, typename U>
inline void itt_store_word_with_release(tbb::atomic<T>& dst, U src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
itt_store_pointer_with_release_v3(&dst, (void *)uintptr_t(src));
#else
dst = src;
#endif 
}

template <typename T>
inline T itt_load_word_with_acquire(const tbb::atomic<T>& src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (push)
#pragma warning (disable: 4311)
#endif
T result = (T)itt_load_pointer_with_acquire_v3(&src);
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
#pragma warning (pop)
#endif
return result;
#else
return src;
#endif 
}

template <typename T>
inline void itt_store_word_with_release(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
itt_store_pointer_with_release_v3(&dst, (void *)src);
#else
__TBB_store_with_release(dst, src); 
#endif 
}

template <typename T>
inline T itt_load_word_with_acquire(const T& src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized");
return (T)itt_load_pointer_with_acquire_v3(&src);
#else
return __TBB_load_with_acquire(src);
#endif 
}

template <typename T>
inline void itt_hide_store_word(T& dst, T src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized");
itt_store_pointer_with_release_v3(&dst, (void *)src);
#else
dst = src;
#endif
}

template <typename T>
inline T itt_hide_load_word(const T& src) {
#if TBB_USE_THREADING_TOOLS
__TBB_ASSERT(sizeof(T) == sizeof(void *), "Type must be word-sized.");
return (T)itt_load_pointer_v3(&src);
#else
return src;
#endif
}

#if TBB_USE_THREADING_TOOLS
inline void call_itt_notify(notify_type t, void *ptr) {
call_itt_notify_v5((int)t, ptr);
}
#else
inline void call_itt_notify(notify_type , void * ) {}
#endif 

} 
} 

#endif 
