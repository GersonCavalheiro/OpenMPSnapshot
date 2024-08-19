
#include <boost/winapi/config.hpp>
#include <boost/thread/detail/config.hpp>

#if defined(BOOST_THREAD_WIN32) && defined(BOOST_THREAD_BUILD_LIB)

#if (defined(__MINGW32__) && !defined(_WIN64)) || defined(__MINGW64__) || (__MINGW64_VERSION_MAJOR)

#include <boost/thread/detail/tss_hooks.hpp>

#include <windows.h>

#include <cstdlib>

namespace boost
{
void tss_cleanup_implemented() {}
}

namespace {
void NTAPI on_tls_callback(void* , DWORD dwReason, PVOID )
{
switch (dwReason)
{
case DLL_THREAD_DETACH:
{
boost::on_thread_exit();
break;
}
}
}
}

#if defined(__MINGW64__) || (__MINGW64_VERSION_MAJOR) || (__MINGW32__) || (__MINGW32_MAJOR_VERSION >3) ||   \
((__MINGW32_MAJOR_VERSION==3) && (__MINGW32_MINOR_VERSION>=18))
extern "C"
{
PIMAGE_TLS_CALLBACK __crt_xl_tls_callback__ __attribute__ ((section(".CRT$XLB"))) = on_tls_callback;
}
#else
extern "C" {

void (* after_ctors )() __attribute__((section(".ctors")))     = boost::on_process_enter;
void (* before_dtors)() __attribute__((section(".dtors")))     = boost::on_thread_exit;
void (* after_dtors )() __attribute__((section(".dtors.zzz"))) = boost::on_process_exit;

ULONG __tls_index__ = 0;
char __tls_end__ __attribute__((section(".tls$zzz"))) = 0;
char __tls_start__ __attribute__((section(".tls"))) = 0;


PIMAGE_TLS_CALLBACK __crt_xl_start__ __attribute__ ((section(".CRT$XLA"))) = 0;
PIMAGE_TLS_CALLBACK __crt_xl_end__ __attribute__ ((section(".CRT$XLZ"))) = 0;
}
extern "C" const IMAGE_TLS_DIRECTORY32 _tls_used __attribute__ ((section(".rdata$T"))) =
{
(DWORD) &__tls_start__,
(DWORD) &__tls_end__,
(DWORD) &__tls_index__,
(DWORD) (&__crt_xl_start__+1),
(DWORD) 0,
(DWORD) 0
};
#endif


#elif  defined(_MSC_VER) && !defined(UNDER_CE)

#include <boost/thread/detail/tss_hooks.hpp>

#include <stdlib.h>

#include <windows.h>




#if (_MSC_VER >= 1500)

extern "C" {
extern BOOL (WINAPI * const _pRawDllMainOrig)(HINSTANCE, DWORD, LPVOID);
extern BOOL (WINAPI * const _pDefaultRawDllMainOrig)(HINSTANCE, DWORD, LPVOID) = NULL;
#if defined (_M_IX86)
#pragma comment(linker, "/alternatename:__pRawDllMainOrig=__pDefaultRawDllMainOrig")
#elif defined (_M_X64) || defined (_M_ARM) || defined (_M_ARM64)
#pragma comment(linker, "/alternatename:_pRawDllMainOrig=_pDefaultRawDllMainOrig")
#else  
#error Unsupported platform
#endif  
}

#endif




#if (_MSC_VER < 1300) || ((_MSC_VER > 1900) && (_MSC_VER < 1910)) 
typedef void ( __cdecl *_PVFV_ )();
typedef void ( __cdecl *_PIFV_ )();
#define INIRETSUCCESS_V
#define INIRETSUCCESS_I
#define PVAPI_V void __cdecl
#define PVAPI_I void __cdecl
#elif (_MSC_VER >= 1910)
typedef void ( __cdecl *_PVFV_ )();
typedef int ( __cdecl *_PIFV_ )();
#define INIRETSUCCESS_V
#define INIRETSUCCESS_I 0
#define PVAPI_V void __cdecl
#define PVAPI_I int __cdecl
#else
typedef int ( __cdecl *_PVFV_ )();
typedef int ( __cdecl *_PIFV_ )();
#define INIRETSUCCESS_V 0
#define INIRETSUCCESS_I 0
#define PVAPI_V int __cdecl
#define PVAPI_I int __cdecl
#endif

typedef void (NTAPI* _TLSCB)(HINSTANCE, DWORD, PVOID);


extern "C"
{
extern DWORD _tls_used; 
extern _TLSCB __xl_a[], __xl_z[];    
}

namespace
{

static PVAPI_I on_tls_prepare();
static PVAPI_V on_process_init();
static PVAPI_V on_process_term();
static void NTAPI on_tls_callback(HINSTANCE, DWORD, PVOID);
}

namespace boost
{


#if (_MSC_VER >= 1400)
#pragma section(".CRT$XIU",long,read)
#pragma section(".CRT$XCU",long,read)
#pragma section(".CRT$XTU",long,read)
#pragma section(".CRT$XLC",long,read)
extern const __declspec(allocate(".CRT$XLC")) _TLSCB p_tls_callback = on_tls_callback;
extern const __declspec(allocate(".CRT$XIU")) _PIFV_ p_tls_prepare = on_tls_prepare;
extern const __declspec(allocate(".CRT$XCU")) _PVFV_ p_process_init = on_process_init;
extern const __declspec(allocate(".CRT$XTU")) _PVFV_ p_process_term = on_process_term;
#else
#if (_MSC_VER >= 1300) 
#   pragma data_seg(push, old_seg)
#endif

#pragma data_seg(".CRT$XIU")
extern const _PIFV_ p_tls_prepare = on_tls_prepare;
#pragma data_seg()


#pragma data_seg(".CRT$XCU")
extern const _PVFV_ p_process_init = on_process_init;
#pragma data_seg()


#pragma data_seg(".CRT$XLB")
extern const _TLSCB p_thread_callback = on_tls_callback;
#pragma data_seg()

#pragma data_seg(".CRT$XTU")
extern const _PVFV_ p_process_term = on_process_term;
#pragma data_seg()
#if (_MSC_VER >= 1300) 
#   pragma data_seg(pop, old_seg)
#endif
#endif
} 

namespace
{
#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4189)
#endif

PVAPI_I on_tls_prepare()
{

DWORD volatile dw = _tls_used;

#if (_MSC_VER < 1300) 
_TLSCB* pfbegin = __xl_a;
_TLSCB* pfend = __xl_z;
_TLSCB* pfdst = pfbegin;


while (pfbegin < pfend)
{
if (*pfbegin != 0)
{
*pfdst = *pfbegin;
++pfdst;
}
++pfbegin;
}

*pfdst = 0;
#endif

return INIRETSUCCESS_I;
}
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

PVAPI_V on_process_init()
{


atexit(boost::on_thread_exit);


boost::on_process_enter();

return INIRETSUCCESS_V;
}

PVAPI_V on_process_term()
{
boost::on_process_exit();
return INIRETSUCCESS_V;
}

void NTAPI on_tls_callback(HINSTANCE , DWORD dwReason, PVOID )
{
switch (dwReason)
{
case DLL_THREAD_DETACH:
boost::on_thread_exit();
break;
}
}

#if (_MSC_VER >= 1500)
BOOL WINAPI dll_callback(HINSTANCE hInstance, DWORD dwReason, LPVOID lpReserved)
#else
BOOL WINAPI dll_callback(HINSTANCE, DWORD dwReason, LPVOID)
#endif
{
switch (dwReason)
{
case DLL_THREAD_DETACH:
boost::on_thread_exit();
break;
case DLL_PROCESS_DETACH:
boost::on_process_exit();
break;
}

#if (_MSC_VER >= 1500)
if( _pRawDllMainOrig )
{
return _pRawDllMainOrig(hInstance, dwReason, lpReserved);
}
#endif
return true;
}
} 

extern "C"
{
extern BOOL (WINAPI * const _pRawDllMain)(HINSTANCE, DWORD, LPVOID)=&dll_callback;
}
namespace boost
{
void tss_cleanup_implemented()
{

}
}

#endif 

#endif 
