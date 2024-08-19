



#ifndef __TBB_tbbmalloc_proxy_H
#define __TBB_tbbmalloc_proxy_H

#if _MSC_VER

#ifdef _DEBUG
#pragma comment(lib, "tbbmalloc_proxy_debug.lib")
#else
#pragma comment(lib, "tbbmalloc_proxy.lib")
#endif

#if defined(_WIN64)
#pragma comment(linker, "/include:__TBB_malloc_proxy")
#else
#pragma comment(linker, "/include:___TBB_malloc_proxy")
#endif

#else


extern "C" void __TBB_malloc_proxy();
struct __TBB_malloc_proxy_caller {
__TBB_malloc_proxy_caller() { __TBB_malloc_proxy(); }
} volatile __TBB_malloc_proxy_helper_object;

#endif 


extern "C" int TBB_malloc_replacement_log(char *** function_replacement_log_ptr);

#endif 
