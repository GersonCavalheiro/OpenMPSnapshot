#ifndef BOOST_CONTAINER_DETAIL_IS_CONTIGUOUS_CONTAINER_HPP
#define BOOST_CONTAINER_DETAIL_IS_CONTIGUOUS_CONTAINER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_FUNCNAME data
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_NS_BEG namespace boost { namespace container { namespace is_contiguous_container_detail {
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_NS_END   }}}
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_MIN 0
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_MAX 0
#include <boost/intrusive/detail/has_member_function_callable_with.hpp>

#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_FUNCNAME back_free_capacity
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_NS_BEG namespace boost { namespace container { namespace back_free_capacity_detail {
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_NS_END   }}}
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_MIN 0
#define BOOST_INTRUSIVE_HAS_MEMBER_FUNCTION_CALLABLE_WITH_MAX 0
#include <boost/intrusive/detail/has_member_function_callable_with.hpp>

namespace boost {
namespace container {
namespace dtl {

template <class Container>
struct is_contiguous_container
{
static const bool value =
boost::container::is_contiguous_container_detail::
has_member_function_callable_with_data<Container>::value && 
boost::container::is_contiguous_container_detail::
has_member_function_callable_with_data<const Container>::value;
};


template < class Container
, bool = boost::container::back_free_capacity_detail::
has_member_function_callable_with_back_free_capacity<const Container>::value>
struct back_free_capacity
{
static typename Container::size_type get(const Container &c)
{  return c.back_free_capacity();  }
};

template < class Container>
struct back_free_capacity<Container, false>
{
static typename Container::size_type get(const Container &c)
{  return c.capacity() - c.size();  }
};

}  
}  
}  

#endif   
