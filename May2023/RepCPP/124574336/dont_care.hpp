
#ifndef BOOST_PROTO_DETAIL_DONT_CARE_HPP_EAN_11_07_2007
#define BOOST_PROTO_DETAIL_DONT_CARE_HPP_EAN_11_07_2007

#include <boost/config.hpp>

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable : 4714) 
#endif

namespace boost { namespace proto
{
namespace detail
{
struct dont_care
{
BOOST_FORCEINLINE dont_care(...);
};
}
}}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
