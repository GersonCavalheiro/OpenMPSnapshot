
#include <boost/winapi/config.hpp>
#include <boost/winapi/dll.hpp>
#include <boost/winapi/time.hpp>
#include <boost/winapi/event.hpp>
#include <boost/winapi/handles.hpp>
#include <boost/winapi/thread_pool.hpp>
#include <cstdlib>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/memory_order.hpp>
#include <boost/atomic/atomic.hpp>
#include <boost/thread/win32/interlocked_read.hpp>
#include <boost/thread/win32/thread_primitives.hpp>

namespace boost {
namespace detail {
namespace win32 {

#if BOOST_USE_WINAPI_VERSION >= BOOST_WINAPI_VERSION_WIN6

BOOST_THREAD_DECL boost::detail::win32::detail::gettickcount64_t gettickcount64 = &::boost::winapi::GetTickCount64;

#else 

namespace {

enum init_state
{
uninitialized = 0,
in_progress,
initialized
};

struct get_tick_count64_state
{
boost::atomic< uint64_t > ticks;
boost::atomic< init_state > init;
boost::winapi::HANDLE_ wait_event;
boost::winapi::HANDLE_ wait_handle;
};

BOOST_ALIGNMENT(64) static get_tick_count64_state g_state;

ticks_type WINAPI get_tick_count64()
{
uint64_t old_state = g_state.ticks.load(boost::memory_order_acquire);

uint32_t new_ticks = boost::winapi::GetTickCount();

uint32_t old_ticks = static_cast< uint32_t >(old_state & UINT64_C(0x00000000ffffffff));
uint64_t new_state = ((old_state & UINT64_C(0xffffffff00000000)) + (static_cast< uint64_t >(new_ticks < old_ticks) << 32)) | static_cast< uint64_t >(new_ticks);

g_state.ticks.store(new_state, boost::memory_order_release);

return new_state;
}

void NTAPI refresh_get_tick_count64(boost::winapi::PVOID_, boost::winapi::BOOLEAN_)
{
get_tick_count64();
}

void cleanup_get_tick_count64()
{
if (g_state.wait_handle)
{
boost::winapi::UnregisterWait(g_state.wait_handle);
g_state.wait_handle = NULL;
}

if (g_state.wait_event)
{
boost::winapi::CloseHandle(g_state.wait_event);
g_state.wait_event = NULL;
}
}

ticks_type WINAPI get_tick_count_init()
{
boost::winapi::HMODULE_ hKernel32 = boost::winapi::GetModuleHandleW(L"kernel32.dll");
if (hKernel32)
{
#if defined(BOOST_GCC) && BOOST_GCC >= 80000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
boost::detail::win32::detail::gettickcount64_t p =
(boost::detail::win32::detail::gettickcount64_t)boost::winapi::get_proc_address(hKernel32, "GetTickCount64");
#if defined(BOOST_GCC) && BOOST_GCC >= 80000
#pragma GCC diagnostic pop
#endif
if (p)
{
boost::detail::interlocked_write_release((void**)&gettickcount64, (void*)p);
return p();
}
}

init_state old_init = uninitialized;
if (g_state.init.compare_exchange_strong(old_init, in_progress, boost::memory_order_acq_rel, boost::memory_order_relaxed))
{
if (!g_state.wait_event)
g_state.wait_event = boost::winapi::create_anonymous_event(NULL, false, false);
if (g_state.wait_event)
{
boost::winapi::BOOL_ res = boost::winapi::RegisterWaitForSingleObject(&g_state.wait_handle, g_state.wait_event, &refresh_get_tick_count64, NULL, 0x7fffffff, boost::winapi::WT_EXECUTEINWAITTHREAD_);
if (res)
{
std::atexit(&cleanup_get_tick_count64);

boost::detail::interlocked_write_release((void**)&gettickcount64, (void*)&get_tick_count64);
g_state.init.store(initialized, boost::memory_order_release);
goto finish;
}
}

g_state.init.store(uninitialized, boost::memory_order_release);
}

finish:
return get_tick_count64();
}

} 

BOOST_THREAD_DECL boost::detail::win32::detail::gettickcount64_t gettickcount64 = &get_tick_count_init;

#endif 

} 
} 
} 
