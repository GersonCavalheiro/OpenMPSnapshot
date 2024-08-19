#ifndef BOOST_GIL_CONCEPTS_IMAGE_VIEW_HPP
#define BOOST_GIL_CONCEPTS_IMAGE_VIEW_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/pixel.hpp>
#include <boost/gil/concepts/pixel_dereference.hpp>
#include <boost/gil/concepts/pixel_iterator.hpp>
#include <boost/gil/concepts/pixel_locator.hpp>
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




template <typename View>
struct RandomAccessNDImageViewConcept
{
void constraints()
{
gil_function_requires<Regular<View>>();

using value_type = typename View::value_type;
using reference = typename View::reference; 
using pointer = typename View::pointer;
using difference_type = typename View::difference_type; 
using const_t = typename View::const_t; 
using point_t = typename View::point_t; 
using locator = typename View::locator; 
using iterator = typename View::iterator;
using const_iterator = typename View::const_iterator;
using reverse_iterator = typename View::reverse_iterator;
using size_type = typename View::size_type;
static const std::size_t N=View::num_dimensions;

gil_function_requires<RandomAccessNDLocatorConcept<locator>>();
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<iterator>>();
gil_function_requires<boost_concepts::RandomAccessTraversalConcept<reverse_iterator>>();

using first_it_type = typename View::template axis<0>::iterator;
using last_it_type = typename View::template axis<N-1>::iterator;
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

point_t p;
locator lc;
iterator it;
reverse_iterator rit;
difference_type d; detail::initialize_it(d); ignore_unused_variable_warning(d);

View(p,lc); 

p = view.dimensions();
lc = view.pixels();
size_type sz = view.size(); ignore_unused_variable_warning(sz);
bool is_contiguous = view.is_1d_traversable();
ignore_unused_variable_warning(is_contiguous);

it = view.begin();
it = view.end();
rit = view.rbegin();
rit = view.rend();

reference r1 = view[d]; ignore_unused_variable_warning(r1); 
reference r2 = view(p); ignore_unused_variable_warning(r2); 

first_it_type fi = view.template axis_iterator<0>(p);
ignore_unused_variable_warning(fi);
last_it_type li = view.template axis_iterator<N-1>(p);
ignore_unused_variable_warning(li);

using deref_t = PixelDereferenceAdaptorArchetype<typename View::value_type>;
using dtype = typename View::template add_deref<deref_t>::type;
}
View view;
};

template <typename View>
struct RandomAccess2DImageViewConcept
{
void constraints()
{
gil_function_requires<RandomAccessNDImageViewConcept<View>>();
static_assert(View::num_dimensions == 2, "");

gil_function_requires<RandomAccess2DLocatorConcept<typename View::locator>>();

using dynamic_x_step_t = typename dynamic_x_step_type<View>::type;
using dynamic_y_step_t = typename dynamic_y_step_type<View>::type;
using transposed_t = typename transposed_type<View>::type;
using x_iterator = typename View::x_iterator;
using y_iterator = typename View::y_iterator;
using x_coord_t = typename View::x_coord_t;
using y_coord_t = typename View::y_coord_t;
using xy_locator = typename View::xy_locator;

x_coord_t xd = 0; ignore_unused_variable_warning(xd);
y_coord_t yd = 0; ignore_unused_variable_warning(yd);
x_iterator xit;
y_iterator yit;
typename View::point_t d;

View(xd, yd, xy_locator()); 

xy_locator lc = view.xy_at(xd, yd);
lc = view.xy_at(d);

typename View::reference r = view(xd, yd);
ignore_unused_variable_warning(r);
xd = view.width();
yd = view.height();

xit = view.x_at(d);
xit = view.x_at(xd,yd);
xit = view.row_begin(xd);
xit = view.row_end(xd);

yit = view.y_at(d);
yit = view.y_at(xd,yd);
yit = view.col_begin(xd);
yit = view.col_end(xd);
}
View view;
};

template <typename View>
struct CollectionImageViewConcept
{
void constraints()
{
using value_type = typename View::value_type;
using iterator = typename View::iterator;
using const_iterator =  typename View::const_iterator;
using reference = typename View::reference;
using const_reference = typename View::const_reference;
using pointer = typename View::pointer;
using difference_type = typename View::difference_type;
using size_type=  typename View::size_type;

iterator i;
i = view1.begin();
i = view2.end();

const_iterator ci;
ci = view1.begin();
ci = view2.end();

size_type s;
s = view1.size();
s = view2.size();
ignore_unused_variable_warning(s);

view1.empty();

view1.swap(view2);
}
View view1;
View view2;
};

template <typename View>
struct ForwardCollectionImageViewConcept
{
void constraints()
{
gil_function_requires<CollectionImageViewConcept<View>>();

using reference = typename View::reference;
using const_reference = typename View::const_reference;

reference r = view.front();
ignore_unused_variable_warning(r);

const_reference cr = view.front();
ignore_unused_variable_warning(cr);
}
View view;
};

template <typename View>
struct ReversibleCollectionImageViewConcept
{
void constraints()
{
gil_function_requires<CollectionImageViewConcept<View>>();

using reverse_iterator = typename View::reverse_iterator;
using reference = typename View::reference;
using const_reference = typename View::const_reference;

reverse_iterator i;
i = view.rbegin();
i = view.rend();

reference r = view.back();
ignore_unused_variable_warning(r);

const_reference cr = view.back();
ignore_unused_variable_warning(cr);
}
View view;
};

template <typename View>
struct ImageViewConcept
{
void constraints()
{
gil_function_requires<RandomAccess2DImageViewConcept<View>>();

gil_function_requires<PixelLocatorConcept<typename View::xy_locator>>();

static_assert(std::is_same<typename View::x_coord_t, typename View::y_coord_t>::value, "");

using coord_t = typename View::coord_t; 
std::size_t num_chan = view.num_channels(); ignore_unused_variable_warning(num_chan);
}
View view;
};

namespace detail {

template <typename View>
struct RandomAccessNDImageViewIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccessNDLocatorIsMutableConcept<typename View::locator>>();

gil_function_requires<detail::RandomAccessIteratorIsMutableConcept<typename View::iterator>>();

gil_function_requires<detail::RandomAccessIteratorIsMutableConcept
<
typename View::reverse_iterator
>>();

gil_function_requires<detail::RandomAccessIteratorIsMutableConcept
<
typename View::template axis<0>::iterator
>>();

gil_function_requires<detail::RandomAccessIteratorIsMutableConcept
<
typename View::template axis<View::num_dimensions - 1>::iterator
>>();

typename View::difference_type diff;
initialize_it(diff);
ignore_unused_variable_warning(diff);

typename View::point_t pt;
typename View::value_type v;
initialize_it(v);

view[diff] = v;
view(pt) = v;
}
View view;
};

template <typename View>
struct RandomAccess2DImageViewIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccessNDImageViewIsMutableConcept<View>>();
typename View::x_coord_t xd = 0; ignore_unused_variable_warning(xd);
typename View::y_coord_t yd = 0; ignore_unused_variable_warning(yd);
typename View::value_type v; initialize_it(v);
view(xd, yd) = v;
}
View view;
};

template <typename View>
struct PixelImageViewIsMutableConcept
{
void constraints()
{
gil_function_requires<detail::RandomAccess2DImageViewIsMutableConcept<View>>();
}
};

} 

template <typename View>
struct MutableRandomAccessNDImageViewConcept
{
void constraints()
{
gil_function_requires<RandomAccessNDImageViewConcept<View>>();
gil_function_requires<detail::RandomAccessNDImageViewIsMutableConcept<View>>();
}
};

template <typename View>
struct MutableRandomAccess2DImageViewConcept
{
void constraints()
{
gil_function_requires<RandomAccess2DImageViewConcept<View>>();
gil_function_requires<detail::RandomAccess2DImageViewIsMutableConcept<View>>();
}
};

template <typename View>
struct MutableImageViewConcept
{
void constraints()
{
gil_function_requires<ImageViewConcept<View>>();
gil_function_requires<detail::PixelImageViewIsMutableConcept<View>>();
}
};

template <typename V1, typename V2>
struct views_are_compatible
: pixels_are_compatible<typename V1::value_type, typename V2::value_type>
{
};

template <typename V1, typename V2>
struct ViewsCompatibleConcept
{
void constraints()
{
static_assert(views_are_compatible<V1, V2>::value, "");
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
