#ifndef BOOST_GIL_CONCEPTS_PIXEL_LOCATOR_HPP
#define BOOST_GIL_CONCEPTS_PIXEL_LOCATOR_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/pixel.hpp>
#include <boost/gil/concepts/pixel_dereference.hpp>
#include <boost/gil/concepts/pixel_iterator.hpp>
#include <boost/gil/concepts/point.hpp>
#include <boost/gil/concepts/detail/utility.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

#if defined(BOOST_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunused-local-typedefs"
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

namespace boost { namespace gil {




template <typename Loc>
struct RandomAccessNDLocatorConcept
{
void constraints()
{
gil_function_requires<Regular<Loc>>();


using value_type = typename Loc::value_type;
ignore_unused_variable_warning(value_type{});

using reference = typename Loc::reference;

using difference_type = typename Loc::difference_type;
ignore_unused_variable_warning(difference_type{});

using cached_location_t = typename Loc::cached_location_t;
ignore_unused_variable_warning(cached_location_t{});

using const_t = typename Loc::const_t;
ignore_unused_variable_warning(const_t{});

using point_t = typename Loc::point_t;
ignore_unused_variable_warning(point_t{});

static std::size_t const N = Loc::num_dimensions; ignore_unused_variable_warning(N);

using first_it_type = typename Loc::template axis<0>::iterator;
using last_it_type = typename Loc::template axis<N-1>::iterator;
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<first_it_type>>();
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<last_it_type>>();

gil_function_requires<PointNDConcept<point_t>>();
static_assert(point_t::num_dimensions == N, "");
static_assert(std::is_same
<
typename std::iterator_traits<first_it_type>::difference_type,
typename point_t::template axis<0>::coord_t
>::value, "");
static_assert(std::is_same
<
typename std::iterator_traits<last_it_type>::difference_type,
typename point_t::template axis<N-1>::coord_t
>::value, "");

difference_type d;
loc += d;
loc -= d;
loc = loc + d;
loc = loc - d;
reference r1 = loc[d];  ignore_unused_variable_warning(r1);
reference r2 = *loc;  ignore_unused_variable_warning(r2);
cached_location_t cl = loc.cache_location(d);  ignore_unused_variable_warning(cl);
reference r3 = loc[d];  ignore_unused_variable_warning(r3);

first_it_type fi = loc.template axis_iterator<0>();
fi = loc.template axis_iterator<0>(d);
last_it_type li = loc.template axis_iterator<N-1>();
li = loc.template axis_iterator<N-1>(d);

using deref_t = PixelDereferenceAdaptorArchetype<typename Loc::value_type>;
using dtype = typename Loc::template add_deref<deref_t>::type;
}
Loc loc;
};

template <typename Loc>
struct RandomAccess2DLocatorConcept
{
void constraints()
{
gil_function_requires<RandomAccessNDLocatorConcept<Loc>>();
static_assert(Loc::num_dimensions == 2, "");

using dynamic_x_step_t = typename dynamic_x_step_type<Loc>::type;
using dynamic_y_step_t = typename dynamic_y_step_type<Loc>::type;
using transposed_t = typename transposed_type<Loc>::type;

using cached_location_t = typename Loc::cached_location_t;
gil_function_requires<Point2DConcept<typename Loc::point_t>>();

using x_iterator = typename Loc::x_iterator;
using y_iterator = typename Loc::y_iterator;
using x_coord_t = typename Loc::x_coord_t;
using y_coord_t = typename Loc::y_coord_t;

x_coord_t xd = 0; ignore_unused_variable_warning(xd);
y_coord_t yd = 0; ignore_unused_variable_warning(yd);

typename Loc::difference_type d;
typename Loc::reference r=loc(xd,yd);  ignore_unused_variable_warning(r);

dynamic_x_step_t loc2(dynamic_x_step_t(), yd);
dynamic_x_step_t loc3(dynamic_x_step_t(), xd, yd);

using dynamic_xy_step_transposed_t = typename dynamic_y_step_type
<
typename dynamic_x_step_type<transposed_t>::type
>::type;
dynamic_xy_step_transposed_t loc4(loc, xd,yd,true);

bool is_contiguous = loc.is_1d_traversable(xd);
ignore_unused_variable_warning(is_contiguous);

loc.y_distance_to(loc, xd);

loc = loc.xy_at(d);
loc = loc.xy_at(xd, yd);

x_iterator xit = loc.x_at(d);
xit = loc.x_at(xd, yd);
xit = loc.x();

y_iterator yit = loc.y_at(d);
yit = loc.y_at(xd, yd);
yit = loc.y();

cached_location_t cl = loc.cache_location(xd, yd);
ignore_unused_variable_warning(cl);
}
Loc loc;
};

template <typename Loc>
struct PixelLocatorConcept
{
void constraints()
{
gil_function_requires<RandomAccess2DLocatorConcept<Loc>>();
gil_function_requires<PixelIteratorConcept<typename Loc::x_iterator>>();
gil_function_requires<PixelIteratorConcept<typename Loc::y_iterator>>();
using coord_t = typename Loc::coord_t;
static_assert(std::is_same<typename Loc::x_coord_t, typename Loc::y_coord_t>::value, "");
}
Loc loc;
};

namespace detail {

template <typename Loc>
struct RandomAccessNDLocatorIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccessIteratorIsMutableConcept
<
typename Loc::template axis<0>::iterator
>>();
gil_function_requires<detail::RandomAccessIteratorIsMutableConcept
<
typename Loc::template axis<Loc::num_dimensions-1>::iterator
>>();

typename Loc::difference_type d; initialize_it(d);
typename Loc::value_type v; initialize_it(v);
typename Loc::cached_location_t cl = loc.cache_location(d);
*loc = v;
loc[d] = v;
loc[cl] = v;
}
Loc loc;
};

template <typename Loc>
struct RandomAccess2DLocatorIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccessNDLocatorIsMutableConcept<Loc>>();
typename Loc::x_coord_t xd = 0; ignore_unused_variable_warning(xd);
typename Loc::y_coord_t yd = 0; ignore_unused_variable_warning(yd);
typename Loc::value_type v; initialize_it(v);
loc(xd, yd) = v;
}
Loc loc;
};

} 

template <typename Loc>
struct MutableRandomAccessNDLocatorConcept
{
void constraints()
{
gil_function_requires<RandomAccessNDLocatorConcept<Loc>>();
gil_function_requires<detail::RandomAccessNDLocatorIsMutableConcept<Loc>>();
}
};

template <typename Loc>
struct MutableRandomAccess2DLocatorConcept
{
void constraints()
{
gil_function_requires<RandomAccess2DLocatorConcept<Loc>>();
gil_function_requires<detail::RandomAccess2DLocatorIsMutableConcept<Loc>>();
}
};

template <typename Loc>
struct MutablePixelLocatorConcept
{
void constraints()
{
gil_function_requires<PixelLocatorConcept<Loc>>();
gil_function_requires<detail::RandomAccess2DLocatorIsMutableConcept<Loc>>();
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
