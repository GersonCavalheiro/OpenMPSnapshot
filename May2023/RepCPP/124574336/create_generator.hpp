
#if !defined(BOOST_SPIRIT_KARMA_CREATE_NOV_21_2009_0340PM)
#define BOOST_SPIRIT_KARMA_CREATE_NOV_21_2009_0340PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/karma/auto/meta_create.hpp>

namespace boost { namespace spirit { namespace result_of
{
template <typename T>
struct create_generator
: spirit::traits::meta_create<karma::domain, T> {};
}}}

namespace boost { namespace spirit { namespace karma
{
template <typename T>
typename result_of::create_generator<T>::type
create_generator()
{
return spirit::traits::meta_create<karma::domain, T>::call();
}
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename T>
struct create_generator_exists
: meta_create_exists<karma::domain, T> {};
}}}

#endif
