
#ifndef BOOST_HEAP_POLICIES_HPP
#define BOOST_HEAP_POLICIES_HPP

#include <boost/concept_check.hpp>
#include <boost/parameter/name.hpp>
#include <boost/parameter/template_keyword.hpp>
#include <boost/parameter/aux_/void.hpp>
#include <boost/parameter/binding.hpp>
#include <boost/parameter/parameters.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/integral_constant.hpp>
#include <boost/type_traits/is_void.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace heap {

#ifndef BOOST_DOXYGEN_INVOKED
BOOST_PARAMETER_TEMPLATE_KEYWORD(allocator)
BOOST_PARAMETER_TEMPLATE_KEYWORD(compare)

namespace tag { struct stable; }

template <bool T>
struct stable:
boost::parameter::template_keyword<tag::stable, boost::integral_constant<bool, T> >
{};

namespace tag { struct mutable_; }

template <bool T>
struct mutable_:
boost::parameter::template_keyword<tag::mutable_, boost::integral_constant<bool, T> >
{};


namespace tag { struct constant_time_size; }

template <bool T>
struct constant_time_size:
boost::parameter::template_keyword<tag::constant_time_size, boost::integral_constant<bool, T> >
{};

namespace tag { struct store_parent_pointer; }

template <bool T>
struct store_parent_pointer:
boost::parameter::template_keyword<tag::store_parent_pointer, boost::integral_constant<bool, T> >
{};

namespace tag { struct arity; }

template <unsigned int T>
struct arity:
boost::parameter::template_keyword<tag::arity, boost::integral_constant<int, T> >
{};

namespace tag { struct objects_per_page; }

template <unsigned int T>
struct objects_per_page:
boost::parameter::template_keyword<tag::objects_per_page, boost::integral_constant<int, T> >
{};

BOOST_PARAMETER_TEMPLATE_KEYWORD(stability_counter_type)

namespace detail {

template <typename bound_args, typename tag_type>
struct has_arg
{
typedef typename boost::parameter::binding<bound_args, tag_type, void>::type type;
static const bool value = !boost::is_void<type>::value;
};

template <typename bound_args>
struct extract_stable
{
static const bool has_stable = has_arg<bound_args, tag::stable>::value;

typedef typename boost::conditional<has_stable,
typename has_arg<bound_args, tag::stable>::type,
boost::false_type
>::type stable_t;

static const bool value = stable_t::value;
};

template <typename bound_args>
struct extract_mutable
{
static const bool has_mutable = has_arg<bound_args, tag::mutable_>::value;

typedef typename boost::conditional<has_mutable,
typename has_arg<bound_args, tag::mutable_>::type,
boost::false_type
>::type mutable_t;

static const bool value = mutable_t::value;
};

}

#else


template <typename T>
struct compare{};


template <bool T>
struct mutable_{};


template <typename T>
struct allocator{};


template <bool T>
struct stable{};


template <typename IntType>
struct stability_counter_type{};


template <bool T>
struct constant_time_size{};


template <bool T>
struct store_parent_pointer{};


template <unsigned int T>
struct arity{};
#endif

} 
} 

#endif 
