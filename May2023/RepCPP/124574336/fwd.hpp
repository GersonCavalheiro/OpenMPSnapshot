

#ifndef BOOST_UNORDERED_FWD_HPP_INCLUDED
#define BOOST_UNORDERED_FWD_HPP_INCLUDED

#include <boost/config.hpp>
#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif

#include <boost/predef.h>

#if defined(BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT)
#elif defined(BOOST_LIBSTDCXX11)
#if BOOST_LIBSTDCXX_VERSION > 40600
#define BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT 1
#endif
#elif BOOST_LIB_STD_CXX
#if BOOST_LIB_STD_CXX >= BOOST_VERSION_NUMBER(3, 0, 0)
#define BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT 1
#endif
#elif defined(BOOST_LIB_STD_DINKUMWARE)
#if BOOST_LIB_STD_DINKUMWARE >= BOOST_VERSION_NUMBER(6, 50, 0) ||              \
BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(17, 0, 0)
#define BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT 1
#endif
#endif

#if !defined(BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT)
#define BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT 0
#endif

#if BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT
#include <utility>
#endif

namespace boost {
namespace unordered {
#if BOOST_UNORDERED_HAVE_PIECEWISE_CONSTRUCT
using std::piecewise_construct_t;
using std::piecewise_construct;
#else
struct piecewise_construct_t
{
};
const piecewise_construct_t piecewise_construct = piecewise_construct_t();
#endif
}
}

#endif
