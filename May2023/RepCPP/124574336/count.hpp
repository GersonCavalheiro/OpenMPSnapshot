
#if !defined(FUSION_COUNT_09162005_0158)
#define FUSION_COUNT_09162005_0158

#include <boost/fusion/support/config.hpp>
#include <boost/config.hpp>
#include <boost/mpl/or.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/fusion/support/detail/access.hpp>

#if defined (BOOST_MSVC)
#  pragma warning(push)
#  pragma warning (disable: 4512) 
#endif

namespace boost { namespace fusion { namespace detail
{ 
template <bool is_convertible>
struct compare_convertible;

template <>
struct compare_convertible<true>
{
template <typename T1, typename T2>
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
static bool
call(T1 const& x, T2 const& y)
{
return x == y;
}
};

template <>
struct compare_convertible<false>
{
template <typename T1, typename T2>
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
static bool
call(T1 const&, T2 const&)
{
return false;
}
};

template <typename T1>
struct count_compare
{
typedef typename detail::call_param<T1>::type param;
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
count_compare(param in_x)
: x(in_x) {}

template <typename T2>
BOOST_CONSTEXPR BOOST_FUSION_GPU_ENABLED
bool
operator()(T2 const& y) const
{
return
compare_convertible<
mpl::or_<
is_convertible<T1, T2>
, is_convertible<T2, T1> 
>::value
>::call(x, y);
}

param x;
};
}}}

#if defined (BOOST_MSVC)
#  pragma warning(pop)
#endif

#endif

