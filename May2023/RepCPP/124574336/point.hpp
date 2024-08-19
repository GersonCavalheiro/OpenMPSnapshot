#ifndef BOOST_GIL_CONCEPTS_POINT_HPP
#define BOOST_GIL_CONCEPTS_POINT_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>

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

template <typename T>
class point;

template <std::size_t K, typename T>
T const& axis_value(point<T> const& p);

template <std::size_t K, typename T>
T& axis_value(point<T>& p);

template <typename P>
struct PointNDConcept
{
void constraints()
{
gil_function_requires<Regular<P>>();

using value_type = typename P::value_type;
ignore_unused_variable_warning(value_type{});

static const std::size_t N = P::num_dimensions;
ignore_unused_variable_warning(N);
using FT = typename P::template axis<0>::coord_t;
using LT = typename P::template axis<N - 1>::coord_t;
FT ft = gil::axis_value<0>(point);
axis_value<0>(point) = ft;
LT lt = axis_value<N - 1>(point);
axis_value<N - 1>(point) = lt;

}
P point;
};

template <typename P>
struct Point2DConcept
{
void constraints()
{
gil_function_requires<PointNDConcept<P>>();
static_assert(P::num_dimensions == 2, "");
point.x = point.y;
point[0] = point[1];
}
P point;
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
