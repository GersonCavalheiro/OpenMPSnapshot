
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_ACTION_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_ACTION_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/match_results.hpp> 

namespace boost { namespace xpressive { namespace detail
{

struct actionable
{
virtual ~actionable() {}
virtual void execute(action_args_type *) const {}

actionable()
: next(0)
{}

actionable const *next;
};

}}} 

#endif
