


#ifndef BOOST_ATOMIC_DETAIL_WAIT_OPERATIONS_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_WAIT_OPERATIONS_HPP_INCLUDED_

#include <boost/atomic/detail/wait_ops_generic.hpp>
#include <boost/atomic/detail/wait_ops_emulated.hpp>

#if !defined(BOOST_ATOMIC_DETAIL_WAIT_BACKEND_GENERIC)
#include BOOST_ATOMIC_DETAIL_WAIT_BACKEND_HEADER(boost/atomic/detail/wait_ops_)
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#endif 
