

#ifndef BOOST_WINAPI_PAGE_PROTECTION_FLAGS_HPP_INCLUDED_
#define BOOST_WINAPI_PAGE_PROTECTION_FLAGS_HPP_INCLUDED_

#include <boost/winapi/basic_types.hpp>
#include <boost/winapi/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace winapi {

#if defined( BOOST_USE_WINDOWS_H )

BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_NOACCESS_ = PAGE_NOACCESS;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_READONLY_ = PAGE_READONLY;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_READWRITE_ = PAGE_READWRITE;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_WRITECOPY_ = PAGE_WRITECOPY;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_GUARD_ = PAGE_GUARD;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_NOCACHE_ = PAGE_NOCACHE;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_WRITECOMBINE_ = PAGE_WRITECOMBINE;

#else 

BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_NOACCESS_ = 0x01;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_READONLY_ = 0x02;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_READWRITE_ = 0x04;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_WRITECOPY_ = 0x08;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_GUARD_ = 0x100;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_NOCACHE_ = 0x200;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_WRITECOMBINE_ = 0x400;

#endif 


BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_EXECUTE_ = 0x10;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_EXECUTE_READ_ = 0x20;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_EXECUTE_READWRITE_ = 0x40;
BOOST_CONSTEXPR_OR_CONST DWORD_ PAGE_EXECUTE_WRITECOPY_ = 0x80;

}
}

#include <boost/winapi/detail/footer.hpp>

#endif 
