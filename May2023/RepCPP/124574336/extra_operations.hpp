


#ifndef BOOST_ATOMIC_DETAIL_EXTRA_OPERATIONS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_EXTRA_OPERATIONS_HPP_INCLUDED_

#include <boost/atomic/detail/extra_ops_generic.hpp>
#include <boost/atomic/detail/extra_ops_emulated.hpp>

#if !defined(BOOST_ATOMIC_DETAIL_EXTRA_BACKEND_GENERIC)
#include BOOST_ATOMIC_DETAIL_EXTRA_BACKEND_HEADER(boost/atomic/detail/extra_ops_)
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#endif 
