
#ifndef BOOST_TYPE_INDEX_RUNTIME_CAST_REFERENCE_CAST_HPP
#define BOOST_TYPE_INDEX_RUNTIME_CAST_REFERENCE_CAST_HPP


#include <boost/core/addressof.hpp>
#include <boost/type_index/runtime_cast/detail/runtime_cast_impl.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

struct bad_runtime_cast : std::exception
{};

template<typename T, typename U>
typename boost::add_reference<T>::type runtime_cast(U& u) {
typedef typename boost::remove_reference<T>::type impl_type;
impl_type* value = detail::runtime_cast_impl<impl_type>(
boost::addressof(u), boost::is_base_and_derived<T, U>());
if(!value)
BOOST_THROW_EXCEPTION(bad_runtime_cast());
return *value;
}

template<typename T, typename U>
typename boost::add_reference<const T>::type runtime_cast(U const& u) {
typedef typename boost::remove_reference<T>::type impl_type;
impl_type* value = detail::runtime_cast_impl<impl_type>(
boost::addressof(u), boost::is_base_and_derived<T, U>());
if(!value)
BOOST_THROW_EXCEPTION(bad_runtime_cast());
return *value;
}

}} 

#endif 
