
#ifndef BOOST_TYPE_INDEX_CTTI_REGISTER_CLASS_HPP
#define BOOST_TYPE_INDEX_CTTI_REGISTER_CLASS_HPP


#include <boost/type_index/ctti_type_index.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex { namespace detail {

template <class T>
inline const ctti_data& ctti_construct_typeid_ref(const T*) BOOST_NOEXCEPT {
return ctti_construct<T>();
}

}}} 

#define BOOST_TYPE_INDEX_REGISTER_CLASS                                                                             \
virtual const boost::typeindex::detail::ctti_data& boost_type_index_type_id_runtime_() const BOOST_NOEXCEPT {   \
return boost::typeindex::detail::ctti_construct_typeid_ref(this);                                           \
}                                                                                                               \


#endif 

