


#ifndef BOOST_ATOMIC_DETAIL_LINK_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_LINK_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if (defined(BOOST_ATOMIC_DYN_LINK) || defined(BOOST_ALL_DYN_LINK)) && \
!defined(BOOST_ATOMIC_STATIC_LINK)

#if defined(BOOST_ATOMIC_SOURCE)
#define BOOST_ATOMIC_DECL BOOST_SYMBOL_EXPORT
#define BOOST_ATOMIC_BUILD_DLL
#else
#define BOOST_ATOMIC_DECL BOOST_SYMBOL_IMPORT
#endif

#endif 

#ifndef BOOST_ATOMIC_DECL
#define BOOST_ATOMIC_DECL
#endif

#if !defined(BOOST_ATOMIC_SOURCE) && !defined(BOOST_ALL_NO_LIB) && \
!defined(BOOST_ATOMIC_NO_LIB)

#define BOOST_LIB_NAME boost_atomic

#if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_ATOMIC_DYN_LINK)
#define BOOST_DYN_LINK
#endif

#include <boost/config/auto_link.hpp>

#endif  

#endif
