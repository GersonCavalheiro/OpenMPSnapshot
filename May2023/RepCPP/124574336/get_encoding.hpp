
#if !defined(BOOST_SPIRIT_GET_ENCODING_JANUARY_13_2009_1255PM)
#define BOOST_SPIRIT_GET_ENCODING_JANUARY_13_2009_1255PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/identity.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost { namespace spirit { namespace detail
{
template <typename Modifiers, typename Encoding>
struct get_implicit_encoding
{

typedef typename
mpl::find_if<
char_encodings,
has_modifier<Modifiers, tag::char_encoding_base<mpl::_1> >
>::type
iter;

typedef typename
mpl::eval_if<
is_same<iter, typename mpl::end<char_encodings>::type>,
mpl::identity<Encoding>,
mpl::deref<iter>
>::type
type;
};

template <typename Modifiers, typename Encoding>
struct get_encoding
{

typedef typename
mpl::find_if<
char_encodings,
has_modifier<Modifiers, tag::char_code<tag::encoding, mpl::_1> >
>::type
iter;

typedef typename
mpl::eval_if<
is_same<iter, typename mpl::end<char_encodings>::type>,
get_implicit_encoding<Modifiers, Encoding>,
mpl::deref<iter>
>::type
type;
};

template <typename Modifiers, typename Encoding, bool case_modifier = false>
struct get_encoding_with_case : mpl::identity<Encoding> {};

template <typename Modifiers, typename Encoding>
struct get_encoding_with_case<Modifiers, Encoding, true>
: get_encoding<Modifiers, Encoding> {};
}}}

#endif
