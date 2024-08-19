#ifndef BOOST_GIL_CONCEPTS_COLOR_BASE_HPP
#define BOOST_GIL_CONCEPTS_COLOR_BASE_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/color.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>

#include <boost/core/ignore_unused.hpp>
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

namespace detail {

template <typename Element, typename Layout, int K>
struct homogeneous_color_base;

} 

template <int K, typename E, typename L, int N>
auto at_c(detail::homogeneous_color_base<E, L, N>& p)
-> typename std::add_lvalue_reference<E>::type;

template <int K, typename E, typename L, int N>
auto at_c(detail::homogeneous_color_base<E, L, N> const& p)
-> typename std::add_lvalue_reference<typename std::add_const<E>::type>::type;

template <typename P, typename C, typename L>
struct packed_pixel;

template <int K, typename P, typename C, typename L>
auto at_c(packed_pixel<P, C, L>& p)
-> typename kth_element_reference_type<packed_pixel<P, C, L>, K>::type;

template <int K, typename P, typename C, typename L>
auto at_c(packed_pixel<P, C, L> const& p)
-> typename kth_element_const_reference_type<packed_pixel<P, C, L>, K>::type;

template <typename B, typename C, typename L, bool M>
struct bit_aligned_pixel_reference;

template <int K, typename B, typename C, typename L, bool M>
inline auto at_c(bit_aligned_pixel_reference<B, C, L, M> const& p)
-> typename kth_element_reference_type
<
bit_aligned_pixel_reference<B, C, L, M>,
K
>::type;

template <int K, typename ColorBase>
auto semantic_at_c(ColorBase& p)
-> typename std::enable_if
<
!std::is_const<ColorBase>::value,
typename kth_semantic_element_reference_type<ColorBase, K>::type
>::type;

template <int K, typename ColorBase>
auto semantic_at_c(ColorBase const& p)
-> typename kth_semantic_element_const_reference_type<ColorBase, K>::type;

template <typename ColorBase>
struct ColorBaseConcept
{
void constraints()
{
gil_function_requires<CopyConstructible<ColorBase>>();
gil_function_requires<EqualityComparable<ColorBase>>();

using color_space_t = typename ColorBase::layout_t::color_space_t;
gil_function_requires<ColorSpaceConcept<color_space_t>>();

using channel_mapping_t = typename ColorBase::layout_t::channel_mapping_t;

static const int num_elements = size<ColorBase>::value;

using TN = typename kth_element_type<ColorBase, num_elements - 1>::type;
using RN = typename kth_element_const_reference_type<ColorBase, num_elements - 1>::type;

RN r = gil::at_c<num_elements - 1>(cb);
boost::ignore_unused(r);

semantic_at_c<0>(cb);
semantic_at_c<num_elements-1>(cb);
}
ColorBase cb;
};

template <typename ColorBase>
struct MutableColorBaseConcept
{
void constraints()
{
gil_function_requires<ColorBaseConcept<ColorBase>>();
gil_function_requires<Assignable<ColorBase>>();
gil_function_requires<Swappable<ColorBase>>();

using R0 = typename kth_element_reference_type<ColorBase, 0>::type;

R0 r = gil::at_c<0>(cb);
gil::at_c<0>(cb) = r;
}
ColorBase cb;
};

template <typename ColorBase>
struct ColorBaseValueConcept
{
void constraints()
{
gil_function_requires<MutableColorBaseConcept<ColorBase>>();
gil_function_requires<Regular<ColorBase>>();
}
};

template <typename ColorBase>
struct HomogeneousColorBaseConcept
{
void constraints()
{
gil_function_requires<ColorBaseConcept<ColorBase>>();

static const int num_elements = size<ColorBase>::value;

using T0 = typename kth_element_type<ColorBase, 0>::type;
using TN = typename kth_element_type<ColorBase, num_elements - 1>::type;

static_assert(std::is_same<T0, TN>::value, "");   

using R0 = typename kth_element_const_reference_type<ColorBase, 0>::type;
R0 r = dynamic_at_c(cb, 0);
boost::ignore_unused(r);
}
ColorBase cb;
};

template <typename ColorBase>
struct MutableHomogeneousColorBaseConcept
{
void constraints()
{
gil_function_requires<ColorBaseConcept<ColorBase>>();
gil_function_requires<HomogeneousColorBaseConcept<ColorBase>>();
using R0 = typename kth_element_reference_type<ColorBase, 0>::type;
R0 r = dynamic_at_c(cb, 0);
boost::ignore_unused(r);
dynamic_at_c(cb, 0) = dynamic_at_c(cb, 0);
}
ColorBase cb;
};

template <typename ColorBase>
struct HomogeneousColorBaseValueConcept
{
void constraints()
{
gil_function_requires<MutableHomogeneousColorBaseConcept<ColorBase>>();
gil_function_requires<Regular<ColorBase>>();
}
};

template <typename ColorBase1, typename ColorBase2>
struct ColorBasesCompatibleConcept
{
void constraints()
{
static_assert(std::is_same
<
typename ColorBase1::layout_t::color_space_t,
typename ColorBase2::layout_t::color_space_t
>::value, "");

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
