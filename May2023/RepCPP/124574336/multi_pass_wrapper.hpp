
#if !defined(BOOST_SPIRIT_ITERATOR_MULTI_PASS_WRAPPER_JUL_12_2009_0914PM)
#define BOOST_SPIRIT_ITERATOR_MULTI_PASS_WRAPPER_JUL_12_2009_0914PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/spirit/home/support/detail/scoped_enum_emulation.hpp>

namespace boost { namespace spirit { namespace traits
{

BOOST_SCOPED_ENUM_START(clear_mode) 
{
clear_if_enabled,
clear_always
};
BOOST_SCOPED_ENUM_END

template <typename Iterator>
void clear_queue(Iterator&
, BOOST_SCOPED_ENUM(clear_mode)  = clear_mode::clear_if_enabled) 
{}

template <typename Iterator>
void inhibit_clear_queue(Iterator&, bool) 
{}

template <typename Iterator>
bool inhibit_clear_queue(Iterator&) 
{ 
return false; 
}

template <typename Iterator>
struct is_multi_pass : mpl::false_ {};

}}}

#endif

