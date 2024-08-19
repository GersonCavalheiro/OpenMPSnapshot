
#ifndef BOOST_INTRUSIVE_PRIORITY_COMPARE_HPP
#define BOOST_INTRUSIVE_PRIORITY_COMPARE_HPP

#include <boost/intrusive/detail/config_begin.hpp>
#include <boost/intrusive/detail/workaround.hpp>
#include <boost/intrusive/intrusive_fwd.hpp>

#include <boost/intrusive/detail/minimal_less_equal_header.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace intrusive {


template<class U>
void priority_order();


template <class T = void>
struct priority_compare
{
typedef T      first_argument_type;
typedef T      second_argument_type;
typedef bool   result_type;

BOOST_INTRUSIVE_FORCEINLINE bool operator()(const T &val, const T &val2) const
{
return priority_order(val, val2);
}
};

template <>
struct priority_compare<void>
{
template<class T, class U>
BOOST_INTRUSIVE_FORCEINLINE bool operator()(const T &t, const U &u) const
{
return priority_order(t, u);
}
};


template<class PrioComp, class T>
struct get_prio_comp
{
typedef PrioComp type;
};


template<class T>
struct get_prio_comp<void, T>
{
typedef ::boost::intrusive::priority_compare<T> type;
};


} 
} 

#include <boost/intrusive/detail/config_end.hpp>

#endif 
