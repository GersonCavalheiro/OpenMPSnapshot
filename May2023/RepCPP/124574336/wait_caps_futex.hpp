


#ifndef BOOST_ATOMIC_DETAIL_WAIT_CAPS_FUTEX_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_WAIT_CAPS_FUTEX_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/capabilities.hpp>
#include <boost/atomic/detail/futex.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if defined(BOOST_ATOMIC_DETAIL_HAS_FUTEX)
#define BOOST_ATOMIC_HAS_NATIVE_INT32_WAIT_NOTIFY BOOST_ATOMIC_INT32_LOCK_FREE
#define BOOST_ATOMIC_HAS_NATIVE_INT32_IPC_WAIT_NOTIFY BOOST_ATOMIC_INT32_LOCK_FREE
#endif 

#endif 
