
#ifndef BOOST_TYPE_INDEX_RUNTIME_CAST_POINTER_CAST_HPP
#define BOOST_TYPE_INDEX_RUNTIME_CAST_POINTER_CAST_HPP

#include <boost/type_index.hpp>
#include <boost/type_index/runtime_cast/detail/runtime_cast_impl.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

template<typename T, typename U>
T runtime_cast(U* u) BOOST_NOEXCEPT {
typedef typename boost::remove_pointer<T>::type impl_type;
return detail::runtime_cast_impl<impl_type>(u, boost::is_base_and_derived<T, U>());
}

template<typename T, typename U>
T runtime_cast(U const* u) BOOST_NOEXCEPT {
typedef typename boost::remove_pointer<T>::type impl_type;
return detail::runtime_cast_impl<impl_type>(u, boost::is_base_and_derived<T, U>());
}

template<typename T, typename U>
T* runtime_pointer_cast(U* u) BOOST_NOEXCEPT {
return detail::runtime_cast_impl<T>(u, boost::is_base_and_derived<T, U>());
}

template<typename T, typename U>
T const* runtime_pointer_cast(U const* u) BOOST_NOEXCEPT {
return detail::runtime_cast_impl<T>(u, boost::is_base_and_derived<T, U>());
}

}} 

#endif 
