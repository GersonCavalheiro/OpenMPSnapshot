#ifndef BOOST_MOVE_DETAIL_ITERATOR_TO_RAW_POINTER_HPP
#define BOOST_MOVE_DETAIL_ITERATOR_TO_RAW_POINTER_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/move/detail/iterator_traits.hpp>
#include <boost/move/detail/to_raw_pointer.hpp>
#include <boost/move/detail/pointer_element.hpp>

namespace boost {
namespace movelib {
namespace detail {

template <class T>
BOOST_MOVE_FORCEINLINE T* iterator_to_pointer(T* i)
{  return i; }

template <class Iterator>
BOOST_MOVE_FORCEINLINE typename boost::movelib::iterator_traits<Iterator>::pointer
iterator_to_pointer(const Iterator &i)
{  return i.operator->();  }

template <class Iterator>
struct iterator_to_element_ptr
{
typedef typename boost::movelib::iterator_traits<Iterator>::pointer  pointer;
typedef typename boost::movelib::pointer_element<pointer>::type      element_type;
typedef element_type* type;
};

}  

template <class Iterator>
BOOST_MOVE_FORCEINLINE typename boost::movelib::detail::iterator_to_element_ptr<Iterator>::type
iterator_to_raw_pointer(const Iterator &i)
{
return ::boost::movelib::to_raw_pointer
(  ::boost::movelib::detail::iterator_to_pointer(i)   );
}

}  
}  

#endif   
