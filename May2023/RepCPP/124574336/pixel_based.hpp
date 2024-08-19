#ifndef BOOST_GIL_CONCEPTS_PIXEL_BASED_HPP
#define BOOST_GIL_CONCEPTS_PIXEL_BASED_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/channel.hpp>
#include <boost/gil/concepts/color.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>

#include <cstddef>

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
struct PixelBasedConcept
{
void constraints()
{
using color_space_t = typename color_space_type<P>::type;
gil_function_requires<ColorSpaceConcept<color_space_t>>();

using channel_mapping_t = typename channel_mapping_type<P>::type ;
gil_function_requires<ChannelMappingConcept<channel_mapping_t>>();

static const bool planar = is_planar<P>::value;
ignore_unused_variable_warning(planar);

static const std::size_t nc = num_channels<P>::value;
ignore_unused_variable_warning(nc);
}
};

template <typename P>
struct HomogeneousPixelBasedConcept
{
void constraints()
{
gil_function_requires<PixelBasedConcept<P>>();

using channel_t = typename channel_type<P>::type;
gil_function_requires<ChannelConcept<channel_t>>();
}
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
