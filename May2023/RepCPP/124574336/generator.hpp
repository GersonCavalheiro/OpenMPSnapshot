
#if !defined(BOOST_SPIRIT_GENERATOR_JANUARY_13_2009_1002AM)
#define BOOST_SPIRIT_GENERATOR_JANUARY_13_2009_1002AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/has_xxx.hpp>
#include <boost/mpl/int.hpp>
#include <boost/spirit/home/karma/domain.hpp>

namespace boost { namespace spirit { namespace karma
{
struct generator_properties
{
enum enum_type {
no_properties = 0,
buffering = 0x01,        
counting = 0x02,         
tracking = 0x04,         
disabling = 0x08,        

countingbuffer = 0x03,   
all_properties = 0x0f    
};
};

template <typename Derived>
struct generator
{
struct generator_id;
typedef mpl::int_<generator_properties::no_properties> properties;
typedef Derived derived_type;
typedef karma::domain domain;




Derived const& derived() const
{
return *static_cast<Derived const*>(this);
}
};

template <typename Derived>
struct primitive_generator : generator<Derived>
{
struct primitive_generator_id;
};

template <typename Derived>
struct nary_generator : generator<Derived>
{
struct nary_generator_id;


};

template <typename Derived>
struct unary_generator : generator<Derived>
{
struct unary_generator_id;


};

template <typename Derived>
struct binary_generator : generator<Derived>
{
struct binary_generator_id;




};

}}}

namespace boost { namespace spirit { namespace traits 
{
namespace detail
{
BOOST_MPL_HAS_XXX_TRAIT_DEF(generator_id)
BOOST_MPL_HAS_XXX_TRAIT_DEF(primitive_generator_id)
BOOST_MPL_HAS_XXX_TRAIT_DEF(nary_generator_id)
BOOST_MPL_HAS_XXX_TRAIT_DEF(unary_generator_id)
BOOST_MPL_HAS_XXX_TRAIT_DEF(binary_generator_id)
}

template <typename T>
struct is_generator : detail::has_generator_id<T> {};

template <typename T>
struct is_primitive_generator : detail::has_primitive_generator_id<T> {};

template <typename T>
struct is_nary_generator : detail::has_nary_generator_id<T> {};

template <typename T>
struct is_unary_generator : detail::has_unary_generator_id<T> {};

template <typename T>
struct is_binary_generator : detail::has_binary_generator_id<T> {};

template <typename T>
struct properties_of : T::properties {};

}}}

#endif
