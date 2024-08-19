
#ifndef BOOST_TYPE_INDEX_RUNTIME_CAST_DETAIL_RUNTIME_CAST_IMPL_HPP
#define BOOST_TYPE_INDEX_RUNTIME_CAST_DETAIL_RUNTIME_CAST_IMPL_HPP


#include <boost/type_index.hpp>
#include <boost/type_traits/integral_constant.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

namespace detail {

template<typename T, typename U>
T* runtime_cast_impl(U* u, boost::true_type) BOOST_NOEXCEPT {
return u;
}

template<typename T, typename U>
T const* runtime_cast_impl(U const* u, boost::true_type) BOOST_NOEXCEPT {
return u;
}

template<typename T, typename U>
T* runtime_cast_impl(U* u, boost::false_type) BOOST_NOEXCEPT {
return const_cast<T*>(static_cast<T const*>(
u->boost_type_index_find_instance_(boost::typeindex::type_id<T>())
));
}

template<typename T, typename U>
T const* runtime_cast_impl(U const* u, boost::false_type) BOOST_NOEXCEPT {
return static_cast<T const*>(u->boost_type_index_find_instance_(boost::typeindex::type_id<T>()));
}

} 

}} 

#endif 
