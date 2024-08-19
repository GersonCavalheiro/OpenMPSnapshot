#ifndef BOOST_GIL_CONCEPTS_IMAGE_HPP
#define BOOST_GIL_CONCEPTS_IMAGE_HPP

#include <boost/gil/concepts/basic.hpp>
#include <boost/gil/concepts/concept_check.hpp>
#include <boost/gil/concepts/fwd.hpp>
#include <boost/gil/concepts/image_view.hpp>
#include <boost/gil/concepts/point.hpp>
#include <boost/gil/detail/mp11.hpp>

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

template <typename Image>
struct RandomAccessNDImageConcept
{
void constraints()
{
gil_function_requires<Regular<Image>>();

using view_t = typename Image::view_t;
gil_function_requires<MutableRandomAccessNDImageViewConcept<view_t>>();

using const_view_t = typename Image::const_view_t;
using pixel_t = typename Image::value_type;
using point_t = typename Image::point_t;
gil_function_requires<PointNDConcept<point_t>>();

const_view_t cv = const_view(image);
ignore_unused_variable_warning(cv);
view_t v = view(image);
ignore_unused_variable_warning(v);

pixel_t fill_value;
point_t pt = image.dimensions();
Image image1(pt);
Image image2(pt, 1);
Image image3(pt, fill_value, 1);
image.recreate(pt);
image.recreate(pt, 1);
image.recreate(pt, fill_value, 1);
}
Image image;
};


template <typename Image>
struct RandomAccess2DImageConcept
{
void constraints()
{
gil_function_requires<RandomAccessNDImageConcept<Image>>();
using x_coord_t = typename Image::x_coord_t;
using y_coord_t = typename Image::y_coord_t;
using value_t = typename Image::value_type;

gil_function_requires<MutableRandomAccess2DImageViewConcept<typename Image::view_t>>();

x_coord_t w=image.width();
y_coord_t h=image.height();
value_t fill_value;
Image im1(w,h);
Image im2(w,h,1);
Image im3(w,h,fill_value,1);
image.recreate(w,h);
image.recreate(w,h,1);
image.recreate(w,h,fill_value,1);
}
Image image;
};

template <typename Image>
struct ImageConcept
{
void constraints()
{
gil_function_requires<RandomAccess2DImageConcept<Image>>();
gil_function_requires<MutableImageViewConcept<typename Image::view_t>>();
using coord_t = typename Image::coord_t;
static_assert(num_channels<Image>::value == mp11::mp_size<typename color_space_type<Image>::type>::value, "");

static_assert(std::is_same<coord_t, typename Image::x_coord_t>::value, "");
static_assert(std::is_same<coord_t, typename Image::y_coord_t>::value, "");
}
Image image;
};

}} 

#if defined(BOOST_CLANG)
#pragma clang diagnostic pop
#endif

#if defined(BOOST_GCC) && (BOOST_GCC >= 40900)
#pragma GCC diagnostic pop
#endif

#endif
