#ifndef BOOST_GIL_CONCEPTS_PIXEL_HPP
#define BOOST_GIL_CONCEPTS_PIXEL_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/channel.hpp>
#include <boost/gil/concepts/color.hpp>
#include <boost/gil/concepts/color_base.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/pixel_based.hpp>
#include <boost/gil/concepts/detail/type_traits.hpp>
#include <boost/gil/detail/mp11.hpp>

#include <cstddef>
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

template <typename P>
struct PixelConcept
{
void constraints()
{
gil_function_requires<ColorBaseConcept<P>>();
gil_function_requires<PixelBasedConcept<P>>();

static_assert(is_pixel<P>::value, "");
static const bool is_mutable = P::is_mutable;
ignore_unused_variable_warning(is_mutable);

using value_type = typename P::value_type;

using reference = typename P::reference;
gil_function_requires<PixelConcept
<
typename detail::remove_const_and_reference<reference>::type
>>();

using const_reference = typename P::const_reference;
gil_function_requires<PixelConcept
<
typename detail::remove_const_and_reference<const_reference>::type
>>();
}
};

template <typename P>
struct MutablePixelConcept
{
void constraints()
{
gil_function_requires<PixelConcept<P>>();
static_assert(P::is_mutable, "");
}
};

template <typename P>
struct HomogeneousPixelConcept
{
void constraints()
{
gil_function_requires<PixelConcept<P>>();
gil_function_requires<HomogeneousColorBaseConcept<P>>();
gil_function_requires<HomogeneousPixelBasedConcept<P>>();
p[0];
}
P p;
};

template <typename P>
struct MutableHomogeneousPixelConcept
{
void constraints()
{
gil_function_requires<HomogeneousPixelConcept<P>>();
gil_function_requires<MutableHomogeneousColorBaseConcept<P>>();
p[0] = v;
v = p[0];
}
typename P::template element_type<P>::type v;
P p;
};

template <typename P>
struct PixelValueConcept
{
void constraints()
{
gil_function_requires<PixelConcept<P>>();
gil_function_requires<Regular<P>>();
}
};

template <typename P>
struct HomogeneousPixelValueConcept
{
void constraints()
{
gil_function_requires<HomogeneousPixelConcept<P>>();
gil_function_requires<Regular<P>>();
static_assert(std::is_same<P, typename P::value_type>::value, "");
}
};

namespace detail {

template <typename P1, typename P2, int K>
struct channels_are_pairwise_compatible
: mp11::mp_and
<
channels_are_pairwise_compatible<P1, P2, K - 1>,
channels_are_compatible
<
typename kth_semantic_element_reference_type<P1, K>::type,
typename kth_semantic_element_reference_type<P2, K>::type
>
>
{
};

template <typename P1, typename P2>
struct channels_are_pairwise_compatible<P1, P2, -1> : std::true_type {};

} 

template <typename P1, typename P2>
struct pixels_are_compatible
: mp11::mp_and
<
typename color_spaces_are_compatible
<
typename color_space_type<P1>::type,
typename color_space_type<P2>::type
>::type,
detail::channels_are_pairwise_compatible
<
P1, P2, num_channels<P1>::value - 1
>
>
{
};

template <typename P1, typename P2>
struct PixelsCompatibleConcept
{
void constraints()
{
static_assert(pixels_are_compatible<P1, P2>::value, "");
}
};

template <typename SrcP, typename DstP>
struct PixelConvertibleConcept
{
void constraints()
{
gil_function_requires<PixelConcept<SrcP>>();
gil_function_requires<MutablePixelConcept<DstP>>();
color_convert(src, dst);
}
SrcP src;
DstP dst;
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
