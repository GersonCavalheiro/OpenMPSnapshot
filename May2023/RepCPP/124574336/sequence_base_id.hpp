
#ifndef BOOST_SPIRIT_SUPPORT_SEQUENCE_BASE_ID_HPP
#define BOOST_SPIRIT_SUPPORT_SEQUENCE_BASE_ID_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/has_xxx.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/spirit/home/support/attributes.hpp>

namespace boost { namespace spirit { namespace traits
{
namespace detail
{
BOOST_MPL_HAS_XXX_TRAIT_DEF(sequence_base_id)
}

template <typename Derived, typename Attribute>
struct pass_attribute<Derived, Attribute,
typename enable_if<detail::has_sequence_base_id<Derived> >::type>
: wrap_if_not_tuple<Attribute> {};

}}}

#endif
