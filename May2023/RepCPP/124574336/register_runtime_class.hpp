
#ifndef BOOST_TYPE_INDEX_RUNTIME_CAST_REGISTER_RUNTIME_CLASS_HPP
#define BOOST_TYPE_INDEX_RUNTIME_CAST_REGISTER_RUNTIME_CLASS_HPP

#include <boost/type_index.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

namespace boost { namespace typeindex {

namespace detail {

template<typename T>
inline type_index runtime_class_construct_type_id(T const*) {
return type_id<T>();
}

} 

}} 


#define BOOST_TYPE_INDEX_CHECK_BASE_(r, data, Base) \
if(void const* ret_val = this->Base::boost_type_index_find_instance_(idx)) return ret_val;


#define BOOST_TYPE_INDEX_REGISTER_RUNTIME_CLASS(base_class_seq)                                                          \
BOOST_TYPE_INDEX_REGISTER_CLASS                                                                                      \
BOOST_TYPE_INDEX_IMPLEMENT_RUNTIME_CAST(base_class_seq)

#define BOOST_TYPE_INDEX_IMPLEMENT_RUNTIME_CAST(base_class_seq)                                                          \
virtual void const* boost_type_index_find_instance_(boost::typeindex::type_index const& idx) const BOOST_NOEXCEPT {  \
if(idx == boost::typeindex::detail::runtime_class_construct_type_id(this))                                       \
return this;                                                                                                 \
BOOST_PP_SEQ_FOR_EACH(BOOST_TYPE_INDEX_CHECK_BASE_, _, base_class_seq)                                          \
return NULL;                                                                                                    \
}

#define BOOST_TYPE_INDEX_NO_BASE_CLASS BOOST_PP_SEQ_NIL

#endif 
