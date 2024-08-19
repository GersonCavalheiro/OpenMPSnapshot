
#ifndef BOOST_MEMORY_ORDER_HPP_INCLUDED
#define BOOST_MEMORY_ORDER_HPP_INCLUDED

#include <boost/config.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
# pragma once
#endif

namespace boost
{


#if !defined(BOOST_NO_CXX11_SCOPED_ENUMS)

enum class memory_order : unsigned int
{
relaxed = 0,
consume = 1,
acquire = 2,
release = 4,
acq_rel = 6, 
seq_cst = 14 
};

BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_relaxed = memory_order::relaxed;
BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_consume = memory_order::consume;
BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_acquire = memory_order::acquire;
BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_release = memory_order::release;
BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_acq_rel = memory_order::acq_rel;
BOOST_INLINE_VARIABLE BOOST_CONSTEXPR_OR_CONST memory_order memory_order_seq_cst = memory_order::seq_cst;

#undef BOOST_MEMORY_ORDER_INLINE_VARIABLE

#else 

enum memory_order
{
memory_order_relaxed = 0,
memory_order_consume = 1,
memory_order_acquire = 2,
memory_order_release = 4,
memory_order_acq_rel = 6, 
memory_order_seq_cst = 14 
};

#endif 

} 

#endif 
