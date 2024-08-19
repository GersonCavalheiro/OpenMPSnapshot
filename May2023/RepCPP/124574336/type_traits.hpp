
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_TYPE_TRAITS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_STATIC_TYPE_TRAITS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <string>
#include <boost/config.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename T>
struct is_static_xpression
: mpl::false_
{
};

template<typename Matcher, typename Next>
struct is_static_xpression<static_xpression<Matcher, Next> >
: mpl::true_
{
};

template<typename Top, typename Next>
struct is_static_xpression<stacked_xpression<Top, Next> >
: mpl::true_
{
};

template<typename BidiIter>
struct is_random
: is_convertible
<
typename iterator_category<BidiIter>::type
, std::random_access_iterator_tag
>
{
};

template<typename Iter>
struct is_string_iterator
: mpl::false_
{
};

template<>
struct is_string_iterator<std::string::iterator>
: mpl::true_
{
};

template<>
struct is_string_iterator<std::string::const_iterator>
: mpl::true_
{
};

#ifndef BOOST_NO_STD_WSTRING
template<>
struct is_string_iterator<std::wstring::iterator>
: mpl::true_
{
};

template<>
struct is_string_iterator<std::wstring::const_iterator>
: mpl::true_
{
};
#endif

template<typename T>
struct is_char
: mpl::false_
{};

template<>
struct is_char<char>
: mpl::true_
{};

template<>
struct is_char<wchar_t>
: mpl::true_
{};

}}} 

#endif
