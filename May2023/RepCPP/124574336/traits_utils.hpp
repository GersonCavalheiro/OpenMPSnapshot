
#ifndef BOOST_XPRESSIVE_DETAIL_UTILITY_TRAITS_UTILS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_UTILITY_TRAITS_UTILS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
# pragma warning(push)
# pragma warning(disable : 4100) 
#endif

#include <string>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/xpressive/detail/utility/algorithm.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename ToChar, typename FromChar, typename Traits>
inline ToChar
char_cast(FromChar from, Traits const &, typename enable_if<is_same<ToChar, FromChar> >::type * = 0)
{
return from;
}

template<typename ToChar, typename FromChar, typename Traits>
inline ToChar
char_cast(FromChar from, Traits const &tr, typename disable_if<is_same<ToChar, FromChar> >::type * = 0)
{
BOOST_MPL_ASSERT((is_same<FromChar, char>));
return tr.widen(from);
}

template<typename Traits>
struct widen_fun
{
typedef typename Traits::char_type result_type;
explicit widen_fun(Traits const &tr)
: traits_(tr)
{}

result_type operator()(char ch) const
{
return this->traits_.widen(ch);
}

Traits const &traits_;
};

template<
typename To
, typename From
, typename ToChar = typename detail::range_data<To>::type
, typename FromChar = typename detail::range_data<From>::type
>
struct string_cast_
{
BOOST_MPL_ASSERT((is_same<FromChar, char>));
typedef To const result_type;
template<typename Traits>
result_type operator()(From const &from, Traits const &tr) const
{
widen_fun<Traits> widen(tr);
To to(
boost::make_transform_iterator(detail::data_begin(from), widen)
, boost::make_transform_iterator(detail::data_end(from), widen)
);
return to;
}
};

template<typename To, typename From, typename Char>
struct string_cast_<To, From, Char, Char>
{
typedef To const result_type;
template<typename Traits>
result_type operator()(From const &from, Traits const &) const
{
To to(detail::data_begin(from), detail::data_end(from));
return to;
}
};

template<typename From, typename Char>
struct string_cast_<From, From, Char, Char>
{
typedef From const &result_type;
template<typename Traits>
result_type operator()(From const &from, Traits const &) const
{
return from;
}
};

template<typename To, typename From, typename Traits>
typename string_cast_<To, From>::result_type
string_cast(From const &from, Traits const &tr)
{
return string_cast_<To, From>()(from, tr);
}

template<typename Char, typename Traits>
inline Char translate(Char ch, Traits const &tr, mpl::false_) 
{
return tr.translate(ch);
}

template<typename Char, typename Traits>
inline Char translate(Char ch, Traits const &tr, mpl::true_) 
{
return tr.translate_nocase(ch);
}

}}} 

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
