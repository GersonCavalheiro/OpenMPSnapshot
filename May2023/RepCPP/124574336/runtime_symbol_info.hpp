
#ifndef BOOST_DLL_RUNTIME_SYMBOL_INFO_HPP
#define BOOST_DLL_RUNTIME_SYMBOL_INFO_HPP

#include <boost/dll/config.hpp>
#include <boost/predef/os.h>
#include <boost/predef/compiler/visualc.h>
#include <boost/dll/detail/aggressive_ptr_cast.hpp>
#if BOOST_OS_WINDOWS
#   include <boost/winapi/dll.hpp>
#   include <boost/dll/detail/windows/path_from_handle.hpp>
#else
#   include <dlfcn.h>
#   include <boost/dll/detail/posix/program_location_impl.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace dll {

#if BOOST_OS_WINDOWS
namespace detail {
inline boost::dll::fs::path program_location_impl(boost::dll::fs::error_code& ec) {
return boost::dll::detail::path_from_handle(NULL, ec);
}
} 
#endif


template <class T>
inline boost::dll::fs::path symbol_location_ptr(T ptr_to_symbol, boost::dll::fs::error_code& ec) {
BOOST_STATIC_ASSERT_MSG(boost::is_pointer<T>::value, "boost::dll::symbol_location_ptr works only with pointers! `ptr_to_symbol` must be a pointer");
boost::dll::fs::path ret;
if (!ptr_to_symbol) {
ec = boost::dll::fs::make_error_code(
boost::dll::fs::errc::bad_address
);

return ret;
}
ec.clear();

const void* ptr = boost::dll::detail::aggressive_ptr_cast<const void*>(ptr_to_symbol);

#if BOOST_OS_WINDOWS
boost::winapi::MEMORY_BASIC_INFORMATION_ mbi;
if (!boost::winapi::VirtualQuery(ptr, &mbi, sizeof(mbi))) {
ec = boost::dll::detail::last_error_code();
return ret;
}

return boost::dll::detail::path_from_handle(reinterpret_cast<boost::winapi::HMODULE_>(mbi.AllocationBase), ec);
#else
Dl_info info;

const int res = dladdr(const_cast<void*>(ptr), &info);

if (res) {
ret = info.dli_fname;
} else {
boost::dll::detail::reset_dlerror();
ec = boost::dll::fs::make_error_code(
boost::dll::fs::errc::bad_address
);
}

return ret;
#endif
}

template <class T>
inline boost::dll::fs::path symbol_location_ptr(T ptr_to_symbol) {
boost::dll::fs::path ret;
boost::dll::fs::error_code ec;
ret = boost::dll::symbol_location_ptr(ptr_to_symbol, ec);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::symbol_location_ptr(T ptr_to_symbol) failed");
}

return ret;
}


template <class T>
inline boost::dll::fs::path symbol_location(const T& symbol, boost::dll::fs::error_code& ec) {
ec.clear();
return boost::dll::symbol_location_ptr(
boost::dll::detail::aggressive_ptr_cast<const void*>(boost::addressof(symbol)),
ec
);
}

#if BOOST_COMP_MSVC < BOOST_VERSION_NUMBER(14,0,0)
template <class T>
inline boost::dll::fs::path symbol_location(const T& symbol, const char*  = 0)
#else
template <class T>
inline boost::dll::fs::path symbol_location(const T& symbol)
#endif
{
boost::dll::fs::path ret;
boost::dll::fs::error_code ec;
ret = boost::dll::symbol_location_ptr(
boost::dll::detail::aggressive_ptr_cast<const void*>(boost::addressof(symbol)),
ec
);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::symbol_location(const T& symbol) failed");
}

return ret;
}

namespace {


static inline boost::dll::fs::path this_line_location(boost::dll::fs::error_code& ec) {
typedef boost::dll::fs::path(func_t)(boost::dll::fs::error_code& );
func_t& f = this_line_location;
return boost::dll::symbol_location(f, ec);
}

static inline boost::dll::fs::path this_line_location() {
boost::dll::fs::path ret;
boost::dll::fs::error_code ec;
ret = this_line_location(ec);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::this_line_location() failed");
}

return ret;
}

} 


inline boost::dll::fs::path program_location(boost::dll::fs::error_code& ec) {
ec.clear();
return boost::dll::detail::program_location_impl(ec);
}

inline boost::dll::fs::path program_location() {
boost::dll::fs::path ret;
boost::dll::fs::error_code ec;
ret = boost::dll::detail::program_location_impl(ec);

if (ec) {
boost::dll::detail::report_error(ec, "boost::dll::program_location() failed");
}

return ret;
}

}} 

#endif 

