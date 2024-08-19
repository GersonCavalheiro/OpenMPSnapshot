
#ifndef BOOST_DLL_SHARED_LIBRARY_MODE_HPP
#define BOOST_DLL_SHARED_LIBRARY_MODE_HPP

#include <boost/dll/config.hpp>
#include <boost/predef/os.h>
#include <boost/predef/library/c.h>

#if BOOST_OS_WINDOWS
#   include <boost/winapi/dll.hpp>
#else
#   include <dlfcn.h>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif


namespace boost { namespace dll { namespace load_mode {



enum type {
#ifdef BOOST_DLL_DOXYGEN

default_mode,


dont_resolve_dll_references,


load_ignore_code_authz_level,


load_with_altered_search_path,


rtld_lazy,


rtld_now,


rtld_global,


rtld_local,


rtld_deepbind,


append_decorations,

search_system_folders
#elif BOOST_OS_WINDOWS
default_mode                          = 0,
dont_resolve_dll_references           = boost::winapi::DONT_RESOLVE_DLL_REFERENCES_,
load_ignore_code_authz_level          = boost::winapi::LOAD_IGNORE_CODE_AUTHZ_LEVEL_,
load_with_altered_search_path         = boost::winapi::LOAD_WITH_ALTERED_SEARCH_PATH_,
rtld_lazy                             = 0,
rtld_now                              = 0,
rtld_global                           = 0,
rtld_local                            = 0,
rtld_deepbind                         = 0,
append_decorations                    = 0x00800000,
search_system_folders                 = (append_decorations << 1)
#else
default_mode                          = 0,
dont_resolve_dll_references           = 0,
load_ignore_code_authz_level          = 0,
load_with_altered_search_path         = 0,
rtld_lazy                             = RTLD_LAZY,
rtld_now                              = RTLD_NOW,
rtld_global                           = RTLD_GLOBAL,
rtld_local                            = RTLD_LOCAL,

#if BOOST_LIB_C_GNU < BOOST_VERSION_NUMBER(2,3,4)
rtld_deepbind                         = 0,
#else
rtld_deepbind                         = RTLD_DEEPBIND,
#endif

append_decorations                    = 0x00800000,
search_system_folders                 = (append_decorations << 1)
#endif
};


BOOST_CONSTEXPR inline type operator|(type left, type right) BOOST_NOEXCEPT {
return static_cast<type>(
static_cast<unsigned int>(left) | static_cast<unsigned int>(right)
);
}
BOOST_CXX14_CONSTEXPR inline type& operator|=(type& left, type right) BOOST_NOEXCEPT {
left = left | right;
return left;
}

BOOST_CONSTEXPR inline type operator&(type left, type right) BOOST_NOEXCEPT {
return static_cast<type>(
static_cast<unsigned int>(left) & static_cast<unsigned int>(right)
);
}
BOOST_CXX14_CONSTEXPR inline type& operator&=(type& left, type right) BOOST_NOEXCEPT {
left = left & right;
return left;
}

BOOST_CONSTEXPR inline type operator^(type left, type right) BOOST_NOEXCEPT {
return static_cast<type>(
static_cast<unsigned int>(left) ^ static_cast<unsigned int>(right)
);
}
BOOST_CXX14_CONSTEXPR inline type& operator^=(type& left, type right) BOOST_NOEXCEPT {
left = left ^ right;
return left;
}

BOOST_CONSTEXPR inline type operator~(type left) BOOST_NOEXCEPT {
return static_cast<type>(
~static_cast<unsigned int>(left)
);
}

}}} 

#endif 
