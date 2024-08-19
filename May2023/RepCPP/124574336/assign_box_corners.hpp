



#ifndef BOOST_GEOMETRY_ALGORITHMS_DETAIL_ASSIGN_BOX_CORNERS_HPP
#define BOOST_GEOMETRY_ALGORITHMS_DETAIL_ASSIGN_BOX_CORNERS_HPP


#include <cstddef>

#include <boost/geometry/geometries/concepts/check.hpp>
#include <boost/geometry/algorithms/detail/assign_values.hpp>
#include <boost/geometry/util/range.hpp>


namespace boost { namespace geometry
{

#ifndef DOXYGEN_NO_DETAIL
namespace detail
{


template <typename Box, typename Point>
inline void assign_box_corners(Box const& box,
Point& lower_left, Point& lower_right,
Point& upper_left, Point& upper_right)
{
concepts::check<Box const>();
concepts::check<Point>();

detail::assign::assign_box_2d_corner
<min_corner, min_corner>(box, lower_left);
detail::assign::assign_box_2d_corner
<max_corner, min_corner>(box, lower_right);
detail::assign::assign_box_2d_corner
<min_corner, max_corner>(box, upper_left);
detail::assign::assign_box_2d_corner
<max_corner, max_corner>(box, upper_right);
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#endif


template <bool Reverse, typename Box, typename Range>
inline void assign_box_corners_oriented(Box const& box, Range& corners)
{
if (Reverse)
{
assign_box_corners(box,
range::at(corners, 0), range::at(corners, 1),
range::at(corners, 3), range::at(corners, 2));
}
else
{
assign_box_corners(box,
range::at(corners, 0), range::at(corners, 3),
range::at(corners, 1), range::at(corners, 2));
}
}
#if defined(_MSC_VER)
#pragma warning(pop)
#endif


} 
#endif 


}} 


#endif 
