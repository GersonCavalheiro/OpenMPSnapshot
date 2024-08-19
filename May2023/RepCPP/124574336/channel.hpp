#ifndef BOOST_GIL_CONCEPTS_CHANNEL_HPP
#define BOOST_GIL_CONCEPTS_CHANNEL_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>

#include <boost/concept_check.hpp>

#include <utility> 
#include <type_traits>

#if defined(BOOST_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-local-typedefs"
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace boost { namespace gil {

template <typename T>
struct channel_traits;

template <typename DstT, typename SrcT>
auto channel_convert(SrcT const& val)
-> typename channel_traits<DstT>::value_type;

template <typename T>
struct ChannelConcept
{
void constraints()
{
gil_function_requires<boost::EqualityComparableConcept<T>>();

using v = typename channel_traits<T>::value_type;
using r = typename channel_traits<T>::reference;
using p = typename channel_traits<T>::pointer;
using cr = typename channel_traits<T>::const_reference;
using cp = typename channel_traits<T>::const_pointer;

channel_traits<T>::min_value();
channel_traits<T>::max_value();
}

T c;
};

namespace detail
{

template <typename T>
struct ChannelIsMutableConcept
{
void constraints()
{
c1 = c2;
using std::swap;
swap(c1, c2);
}
T c1;
T c2;
};

} 

template <typename T>
struct MutableChannelConcept
{
void constraints()
{
gil_function_requires<ChannelConcept<T>>();
gil_function_requires<detail::ChannelIsMutableConcept<T>>();
}
};

template <typename T>
struct ChannelValueConcept
{
void constraints()
{
gil_function_requires<ChannelConcept<T>>();
gil_function_requires<Regular<T>>();
}
};

template <typename T1, typename T2>  
struct channels_are_compatible
: std::is_same
<
typename channel_traits<T1>::value_type,
typename channel_traits<T2>::value_type
>
{
};

template <typename Channel1, typename Channel2>
struct ChannelsCompatibleConcept
{
void constraints()
{
static_assert(channels_are_compatible<Channel1, Channel2>::value, "");
}
};

template <typename SrcChannel, typename DstChannel>
struct ChannelConvertibleConcept
{
void constraints()
{
gil_function_requires<ChannelConcept<SrcChannel>>();
gil_function_requires<MutableChannelConcept<DstChannel>>();
dst = channel_convert<DstChannel, SrcChannel>(src);
ignore_unused_variable_warning(dst);
}
SrcChannel src;
DstChannel dst;
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
