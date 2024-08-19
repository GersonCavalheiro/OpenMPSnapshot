#ifndef BOOST_CORE_NO_EXCEPTIONS_SUPPORT_HPP
#define BOOST_CORE_NO_EXCEPTIONS_SUPPORT_HPP

#if defined(_MSC_VER)
#  pragma once
#endif


#include <boost/config.hpp>
#include <boost/config/workaround.hpp>

#if !(defined BOOST_NO_EXCEPTIONS)
#    define BOOST_TRY { try
#    define BOOST_CATCH(x) catch(x)
#    define BOOST_RETHROW throw;
#    define BOOST_CATCH_END }
#else
#    if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
#        define BOOST_TRY { if ("")
#        define BOOST_CATCH(x) else if (!"")
#    elif !defined(BOOST_MSVC) || BOOST_MSVC >= 1900
#        define BOOST_TRY { if (true)
#        define BOOST_CATCH(x) else if (false)
#    else
#        define BOOST_TRY { \
__pragma(warning(push)) \
__pragma(warning(disable: 4127)) \
if (true) \
__pragma(warning(pop))
#        define BOOST_CATCH(x) else \
__pragma(warning(push)) \
__pragma(warning(disable: 4127)) \
if (false) \
__pragma(warning(pop))
#    endif
#    define BOOST_RETHROW
#    define BOOST_CATCH_END }
#endif


#endif 
