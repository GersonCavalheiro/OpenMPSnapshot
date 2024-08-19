
#ifndef BOOST_TYPE_INDEX_STL_REGISTER_CLASS_HPP
#define BOOST_TYPE_INDEX_STL_REGISTER_CLASS_HPP


#include <boost/type_index/stl_type_index.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex { namespace detail {

template <class T>
inline const stl_type_index::type_info_t& stl_construct_typeid_ref(const T*) BOOST_NOEXCEPT {
return typeid(T);
}

}}} 

#define BOOST_TYPE_INDEX_REGISTER_CLASS                                                                                     \
virtual const boost::typeindex::stl_type_index::type_info_t& boost_type_index_type_id_runtime_() const BOOST_NOEXCEPT { \
return boost::typeindex::detail::stl_construct_typeid_ref(this);                                                    \
}                                                                                                                       \


#endif 

