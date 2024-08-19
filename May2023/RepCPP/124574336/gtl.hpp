
#ifndef BOOST_POLYGON_GTL_HPP
#define BOOST_POLYGON_GTL_HPP

#ifdef __ICC
#pragma warning (push)
#pragma warning (disable:1125)
#endif

#ifdef WIN32
#pragma warning (push)
#pragma warning( disable: 4996 )
#pragma warning( disable: 4800 )
#endif

#define BOOST_POLYGON_NO_DEPS
#include "polygon.hpp"
namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

#ifdef WIN32
#pragma warning (pop)
#endif

#ifdef __ICC
#pragma warning (pop)
#endif

#endif
