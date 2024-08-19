#ifndef BOOST_GIL_CONCEPTS_COLOR_HPP
#define BOOST_GIL_CONCEPTS_COLOR_HPP

#include <boost/gil/concepts/concept_check.hpp>

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

template <typename CS>
struct ColorSpaceConcept
{
void constraints()
{

}
};

template <typename CS1, typename CS2>
struct color_spaces_are_compatible : std::is_same<CS1, CS2> {};

template <typename CS1, typename CS2>
struct ColorSpacesCompatibleConcept
{
void constraints()
{
static_assert(color_spaces_are_compatible<CS1, CS2>::value, "");
}
};

template <typename CM>
struct ChannelMappingConcept
{
void constraints()
{

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
