
#ifndef BOOST_STACKTRACE_DETAIL_VOID_PTR_CAST_HPP
#define BOOST_STACKTRACE_DETAIL_VOID_PTR_CAST_HPP

#include <boost/config.hpp>
#ifdef BOOST_HAS_PRAGMA_ONCE
#   pragma once
#endif

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_pointer.hpp>

#if defined(__GNUC__) && defined(__GNUC_MINOR__) && (__GNUC__ * 100 + __GNUC_MINOR__ > 301)
#   pragma GCC system_header
#endif

namespace boost { namespace stacktrace { namespace detail {

template <class To, class From>
To void_ptr_cast(From* v) BOOST_NOEXCEPT {
BOOST_STATIC_ASSERT_MSG(
boost::is_pointer<To>::value,
"`void_ptr_cast` function must be used only for casting to or from void pointers."
);

BOOST_STATIC_ASSERT_MSG(
sizeof(From*) == sizeof(To),
"Pointer to function and pointer to object differ in size on your platform."
);

return reinterpret_cast<To>(v);
}


}}} 

#endif 

