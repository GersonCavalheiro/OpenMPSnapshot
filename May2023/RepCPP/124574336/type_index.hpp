
#ifndef BOOST_TYPE_INDEX_HPP
#define BOOST_TYPE_INDEX_HPP


#include <boost/config.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
# pragma once
#endif

#if defined(BOOST_TYPE_INDEX_USER_TYPEINDEX)
#   include BOOST_TYPE_INDEX_USER_TYPEINDEX
#   ifdef BOOST_HAS_PRAGMA_DETECT_MISMATCH
#       pragma detect_mismatch( "boost__type_index__abi", "user defined type_index class is used: " BOOST_STRINGIZE(BOOST_TYPE_INDEX_USER_TYPEINDEX))
#   endif
#elif (!defined(BOOST_NO_RTTI) && !defined(BOOST_TYPE_INDEX_FORCE_NO_RTTI_COMPATIBILITY)) || defined(BOOST_MSVC)
#   include <boost/type_index/stl_type_index.hpp>
#   if defined(BOOST_NO_RTTI) || defined(BOOST_TYPE_INDEX_FORCE_NO_RTTI_COMPATIBILITY)
#       include <boost/type_index/detail/stl_register_class.hpp>
#       ifdef BOOST_HAS_PRAGMA_DETECT_MISMATCH
#           pragma detect_mismatch( "boost__type_index__abi", "RTTI is off - typeid() is used only for templates")
#       endif
#   else
#       ifdef BOOST_HAS_PRAGMA_DETECT_MISMATCH
#           pragma detect_mismatch( "boost__type_index__abi", "RTTI is used")
#       endif
#   endif
#else
#   include <boost/type_index/ctti_type_index.hpp>
#   include <boost/type_index/detail/ctti_register_class.hpp>
#   ifdef BOOST_HAS_PRAGMA_DETECT_MISMATCH
#       pragma detect_mismatch( "boost__type_index__abi", "RTTI is off - using CTTI")
#   endif
#endif

#ifndef BOOST_TYPE_INDEX_REGISTER_CLASS
#define BOOST_TYPE_INDEX_REGISTER_CLASS
#endif

namespace boost { namespace typeindex {

#if defined(BOOST_TYPE_INDEX_DOXYGEN_INVOKED)

#define BOOST_TYPE_INDEX_FUNCTION_SIGNATURE BOOST_CURRENT_FUNCTION

#define BOOST_TYPE_INDEX_CTTI_USER_DEFINED_PARSING (0, 0, false, "")


typedef platform_specific type_index;
#elif defined(BOOST_TYPE_INDEX_USER_TYPEINDEX)
#elif (!defined(BOOST_NO_RTTI) && !defined(BOOST_TYPE_INDEX_FORCE_NO_RTTI_COMPATIBILITY)) || defined(BOOST_MSVC)
typedef boost::typeindex::stl_type_index type_index;
#else 
typedef boost::typeindex::ctti_type_index type_index;
#endif

typedef type_index::type_info_t type_info;

#if defined(BOOST_TYPE_INDEX_DOXYGEN_INVOKED)

#define BOOST_TYPE_INDEX_USER_TYPEINDEX <full/absolute/path/to/header/with/type_index.hpp>


#define BOOST_TYPE_INDEX_REGISTER_CLASS nothing-or-some-virtual-functions

#define BOOST_TYPE_INDEX_FORCE_NO_RTTI_COMPATIBILITY

#endif 


template <class T>
inline type_index type_id() BOOST_NOEXCEPT {
return type_index::type_id<T>();
}

template <class T>
inline type_index type_id_with_cvr() BOOST_NOEXCEPT {
return type_index::type_id_with_cvr<T>();
}

template <class T>
inline type_index type_id_runtime(const T& runtime_val) BOOST_NOEXCEPT {
return type_index::type_id_runtime(runtime_val);
}

}} 



#endif 

