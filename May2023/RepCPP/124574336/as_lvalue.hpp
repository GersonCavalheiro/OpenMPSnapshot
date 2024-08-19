
#ifndef BOOST_PROTO_TRANSFORM_AS_LVALUE_HPP_EAN_12_27_2007
#define BOOST_PROTO_TRANSFORM_AS_LVALUE_HPP_EAN_12_27_2007

#include <boost/proto/proto_fwd.hpp>

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable : 4714) 
#endif

namespace boost { namespace proto
{
namespace detail
{
template<typename T>
BOOST_FORCEINLINE
T &as_lvalue(T &t)
{
return t;
}

template<typename T>
BOOST_FORCEINLINE
T const &as_lvalue(T const &t)
{
return t;
}
}
}}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
