




#ifndef BOOST_GEOMETRY_ALGORITHMS_DETAIL_OVERLAY_GET_TURN_INFO_HPP
#define BOOST_GEOMETRY_ALGORITHMS_DETAIL_OVERLAY_GET_TURN_INFO_HPP


#include <boost/core/ignore_unused.hpp>
#include <boost/throw_exception.hpp>

#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/assert.hpp>
#include <boost/geometry/core/config.hpp>
#include <boost/geometry/core/exception.hpp>

#include <boost/geometry/algorithms/convert.hpp>
#include <boost/geometry/algorithms/detail/overlay/get_distance_measure.hpp>
#include <boost/geometry/algorithms/detail/overlay/turn_info.hpp>

#include <boost/geometry/geometries/segment.hpp>

#include <boost/geometry/policies/robustness/robust_point_type.hpp>
#include <boost/geometry/algorithms/detail/overlay/get_turn_info_helpers.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#endif


namespace boost { namespace geometry
{

#if ! defined(BOOST_GEOMETRY_OVERLAY_NO_THROW)
class turn_info_exception : public geometry::exception
{
std::string message;
public:

inline turn_info_exception(char const method)
{
message = "Boost.Geometry Turn exception: ";
message += method;
}

virtual ~turn_info_exception() throw()
{}

virtual char const* what() const throw()
{
return message.c_str();
}
};
#endif

#ifndef DOXYGEN_NO_DETAIL
namespace detail { namespace overlay
{

struct base_turn_handler
{
static inline bool opposite(int side1, int side2)
{
return side1 * side2 == -1;
}

static inline bool same(int side1, int side2)
{
return side1 * side2 == 1;
}

template <typename TurnInfo>
static inline void both(TurnInfo& ti, operation_type const op)
{
ti.operations[0].operation = op;
ti.operations[1].operation = op;
}

template <typename TurnInfo>
static inline void ui_else_iu(bool condition, TurnInfo& ti)
{
ti.operations[0].operation = condition
? operation_union : operation_intersection;
ti.operations[1].operation = condition
? operation_intersection : operation_union;
}

template <typename TurnInfo>
static inline void uu_else_ii(bool condition, TurnInfo& ti)
{
both(ti, condition ? operation_union : operation_intersection);
}


#if ! defined(BOOST_GEOMETRY_USE_RESCALING)
template
<
typename UniqueSubRange1,
typename UniqueSubRange2
>
static inline int side_with_distance_measure(UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
int range_index, int point_index)
{
if (range_index >= 1 && range_p.is_last_segment())
{
return 0;
}
if (point_index >= 2 && range_q.is_last_segment())
{
return 0;
}

typedef typename select_coordinate_type
<
typename UniqueSubRange1::point_type,
typename UniqueSubRange2::point_type
>::type coordinate_type;

typedef detail::distance_measure<coordinate_type> dm_type;

dm_type const dm = get_distance_measure(range_p.at(range_index), range_p.at(range_index + 1), range_q.at(point_index));
return dm.measure == 0 ? 0 : dm.measure > 0 ? 1 : -1;
}

template
<
typename UniqueSubRange1,
typename UniqueSubRange2
>
static inline int verified_side(int side,
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
int range_index,
int point_index)
{
return side == 0 ? side_with_distance_measure(range_p, range_q, range_index, point_index) : side;
}
#else
template <typename T1, typename T2>
static inline int verified_side(int side, T1 const& , T2 const& , int , int)
{
return side;
}
#endif


template <typename TurnInfo, typename IntersectionInfo>
static inline void assign_point(TurnInfo& ti,
method_type method,
IntersectionInfo const& info, unsigned int index)
{
ti.method = method;

BOOST_GEOMETRY_ASSERT(index < info.count);

geometry::convert(info.intersections[index], ti.point);
ti.operations[0].fraction = info.fractions[index].robust_ra;
ti.operations[1].fraction = info.fractions[index].robust_rb;
}

template <typename TurnInfo, typename IntersectionInfo, typename DirInfo>
static inline void assign_point_and_correct(TurnInfo& ti,
method_type method,
IntersectionInfo const& info, DirInfo const& dir_info)
{
ti.method = method;

static int const index = 0;

geometry::convert(info.intersections[index], ti.point);

for (int i = 0; i < 2; i++)
{
if (dir_info.arrival[i] == 1)
{
ti.operations[i].fraction = {1, 1};
}
else if (dir_info.arrival[i] == -1)
{
ti.operations[i].fraction = {0, 1};
}
else
{
ti.operations[i].fraction = i == 0 ? info.fractions[index].robust_ra
: info.fractions[index].robust_rb;
}
}
}

template <typename IntersectionInfo>
static inline unsigned int non_opposite_to_index(IntersectionInfo const& info)
{
return info.fractions[0].robust_rb < info.fractions[1].robust_rb
? 1 : 0;
}

template <typename Point1, typename Point2>
static inline typename geometry::coordinate_type<Point1>::type
distance_measure(Point1 const& a, Point2 const& b)
{
typedef typename geometry::coordinate_type<Point1>::type ctype;
ctype const dx = get<0>(a) - get<0>(b);
ctype const dy = get<1>(a) - get<1>(b);
return dx * dx + dy * dy;
}

template
<
std::size_t IndexP,
std::size_t IndexQ,
typename UniqueSubRange1,
typename UniqueSubRange2,
typename UmbrellaStrategy,
typename TurnInfo
>
static inline void both_collinear(
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
UmbrellaStrategy const&,
std::size_t index_p, std::size_t index_q,
TurnInfo& ti)
{
boost::ignore_unused(range_p, range_q);
BOOST_GEOMETRY_ASSERT(IndexP + IndexQ == 1);
BOOST_GEOMETRY_ASSERT(index_p > 0 && index_p <= 2);
BOOST_GEOMETRY_ASSERT(index_q > 0 && index_q <= 2);

#if ! defined(BOOST_GEOMETRY_USE_RESCALING)
ti.operations[IndexP].remaining_distance = distance_measure(ti.point, range_p.at(index_p));
ti.operations[IndexQ].remaining_distance = distance_measure(ti.point, range_q.at(index_q));

typedef detail::distance_measure
<
typename select_coordinate_type
<
typename UniqueSubRange1::point_type,
typename UniqueSubRange2::point_type
>::type
> dm_type;

const bool p_closer =
ti.operations[IndexP].remaining_distance
<  ti.operations[IndexQ].remaining_distance;
dm_type const dm
= p_closer
? get_distance_measure(range_q.at(index_q - 1),
range_q.at(index_q), range_p.at(index_p))
: get_distance_measure(range_p.at(index_p - 1),
range_p.at(index_p), range_q.at(index_q));

if (! dm.is_zero())
{

bool p_left = p_closer ? dm.is_positive() : dm.is_negative();

ti.operations[IndexP].operation = p_left
? operation_union : operation_intersection;
ti.operations[IndexQ].operation = p_left
? operation_intersection : operation_union;
return;
}
#endif

both(ti, operation_continue);
}

};


template
<
typename TurnInfo
>
struct touch_interior : public base_turn_handler
{

template
<
typename IntersectionInfo,
typename UniqueSubRange
>
static bool handle_as_touch(IntersectionInfo const& info,
UniqueSubRange const& non_touching_range)
{
#if defined(BOOST_GEOMETRY_USE_RESCALING)
return false;
#endif

typedef typename geometry::coordinate_type
<
typename UniqueSubRange::point_type
>::type coor_t;

coor_t const location = distance_measure(info.intersections[0], non_touching_range.at(1));
coor_t const zero = 0;
bool const result = math::equals(location, zero);
return result;
}

template
<
unsigned int Index,
typename UniqueSubRange1,
typename UniqueSubRange2,
typename IntersectionInfo,
typename DirInfo,
typename SidePolicy,
typename UmbrellaStrategy
>
static inline void apply(UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
TurnInfo& ti,
IntersectionInfo const& intersection_info,
DirInfo const& dir_info,
SidePolicy const& side,
UmbrellaStrategy const& umbrella_strategy)
{
assign_point_and_correct(ti, method_touch_interior, intersection_info, dir_info);


BOOST_STATIC_ASSERT(Index <= 1);
static unsigned int const index_p = Index;
static unsigned int const index_q = 1 - Index;

bool const has_pk = ! range_p.is_last_segment();
bool const has_qk = ! range_q.is_last_segment();
int const side_qi_p = dir_info.sides.template get<index_q, 0>();
int const side_qk_p = has_qk ? side.qk_wrt_p1() : 0;

if (side_qi_p == -side_qk_p)
{
unsigned int index = side_qk_p == -1 ? index_p : index_q;
ti.operations[index].operation = operation_union;
ti.operations[1 - index].operation = operation_intersection;
return;
}

int const side_qk_q = has_qk ? side.qk_wrt_q1() : 0;

int const side_pj_q2 = has_qk ? side.pj_wrt_q2() : 0;

if (side_qi_p == -1 && side_qk_p == -1 && side_qk_q == 1)
{
both(ti, operation_intersection);
ti.touch_only = true;
}
else if (side_qi_p == 1 && side_qk_p == 1 && side_qk_q == -1)
{
if (has_qk && side_pj_q2 == -1)
{
both(ti, operation_union);
}
else
{
ti.operations[index_p].operation = operation_union;
ti.operations[index_q].operation = operation_blocked;
}
ti.touch_only = true;
}
else if (side_qi_p == side_qk_p && side_qi_p == side_qk_q)
{
unsigned int index = side_qk_q == 1 ? index_q : index_p;
if (has_qk && side_pj_q2 == 0)
{
index = 1 - index;
}

if (has_pk && has_qk && opposite(side_pj_q2, side_qi_p))
{
int const side_qj_p1 = side.qj_wrt_p1();
int const side_qj_p2 = side.qj_wrt_p2();

if (same(side_qj_p1, side_qj_p2))
{
int const side_pj_q1 = side.pj_wrt_q1();
if (opposite(side_pj_q1, side_pj_q2))
{
index = 1 - index;
}
}
}

ti.operations[index].operation = operation_union;
ti.operations[1 - index].operation = operation_intersection;
ti.touch_only = true;
}
else if (side_qk_p == 0)
{
if (side_qk_q == side_qi_p)
{
both_collinear<index_p, index_q>(range_p, range_q, umbrella_strategy, 1, 2, ti);
}
else
{
ti.operations[index_p].operation = side_qk_q == 1
? operation_intersection
: operation_union;
ti.operations[index_q].operation = operation_blocked;
}
}
else
{
ti.method = method_error;
}
}
};


template
<
typename TurnInfo
>
struct touch : public base_turn_handler
{
static inline bool between(int side1, int side2, int turn)
{
return side1 == side2 && ! opposite(side1, turn);
}

#if ! defined(BOOST_GEOMETRY_USE_RESCALING)
template
<
typename UniqueSubRange1,
typename UniqueSubRange2
>
static inline bool handle_imperfect_touch(UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q, TurnInfo& ti)
{

typedef typename select_coordinate_type
<
typename UniqueSubRange1::point_type,
typename UniqueSubRange2::point_type
>::type coordinate_type;

typedef detail::distance_measure<coordinate_type> dm_type;

dm_type const dm_qj_p1 = get_distance_measure(range_p.at(0), range_p.at(1), range_q.at(1));
dm_type const dm_pi_q2 = get_distance_measure(range_q.at(1), range_q.at(2), range_p.at(0));

if (dm_qj_p1.measure > 0 && dm_pi_q2.measure > 0)
{
ti.operations[0].operation = operation_blocked;
ti.operations[1].operation = operation_union;
ti.touch_only = true;
return true;
}

dm_type const dm_pj_q1 = get_distance_measure(range_q.at(0), range_q.at(1), range_p.at(1));
dm_type const dm_qi_p2 = get_distance_measure(range_p.at(1), range_p.at(2), range_q.at(0));

if (dm_pj_q1.measure > 0 && dm_qi_p2.measure > 0)
{
ti.operations[0].operation = operation_union;
ti.operations[1].operation = operation_blocked;
ti.touch_only = true;
return true;
}
return false;
}
#endif

template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename IntersectionInfo,
typename DirInfo,
typename SideCalculator,
typename UmbrellaStrategy
>
static inline void apply(UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
TurnInfo& ti,
IntersectionInfo const& intersection_info,
DirInfo const& dir_info,
SideCalculator const& side,
UmbrellaStrategy const& umbrella_strategy)
{
assign_point_and_correct(ti, method_touch, intersection_info, dir_info);

bool const has_pk = ! range_p.is_last_segment();
bool const has_qk = ! range_q.is_last_segment();

int const side_pk_q1 = has_pk ? side.pk_wrt_q1() : 0;

int const side_qi_p1 = verified_side(dir_info.sides.template get<1, 0>(), range_p, range_q, 0, 0);
int const side_qk_p1 = has_qk ? verified_side(side.qk_wrt_p1(), range_p, range_q, 0, 2) : 0;

if (! opposite(side_qi_p1, side_qk_p1))
{
int const side_pk_q2 = has_pk && has_qk ? side.pk_wrt_q2() : 0;
int const side_pk_p  = has_pk ? side.pk_wrt_p1() : 0;
int const side_qk_q  = has_qk ? side.qk_wrt_q1() : 0;

bool const q_turns_left = side_qk_q == 1;

bool const block_q = side_qk_p1 == 0
&& ! same(side_qi_p1, side_qk_q)
;

if (side_pk_p == side_qi_p1
|| side_pk_p == side_qk_p1
|| (side_qi_p1 == 0 && side_qk_p1 == 0 && side_pk_p != -1))
{
#if ! defined(BOOST_GEOMETRY_USE_RESCALING)
if (side_qk_p1 == 0 && side_pk_q1 == 0
&& has_qk && has_qk
&& handle_imperfect_touch(range_p, range_q, ti))
{
return;
}
#endif
if (side_pk_q2 == 0 && ! block_q)
{
both_collinear<0, 1>(range_p, range_q, umbrella_strategy, 2, 2, ti);
return;
}

if (side_pk_q1 == 0)
{
ti.operations[0].operation = operation_blocked;
ti.operations[1].operation = block_q ? operation_blocked
: q_turns_left ? operation_intersection
: operation_union;
return;
}

if (between(side_pk_q1, side_pk_q2, side_qk_q))
{
ui_else_iu(q_turns_left, ti);
if (block_q)
{
ti.operations[1].operation = operation_blocked;
}
return;
}

if (side_pk_q2 == -side_qk_q)
{
ui_else_iu(! q_turns_left, ti);
ti.touch_only = true;
return;
}

if (side_pk_q1 == -side_qk_q)
{
uu_else_ii(! q_turns_left, ti);
if (block_q)
{
ti.operations[1].operation = operation_blocked;
}
else
{
ti.touch_only = true;
}
return;
}
}
else
{
ti.operations[0].operation = q_turns_left
? operation_intersection
: operation_union;
ti.operations[1].operation = block_q
? operation_blocked
: side_qi_p1 == 1 || side_qk_p1 == 1
? operation_union
: operation_intersection;
if (! block_q)
{
ti.touch_only = true;
}

return;
}
}
else
{
int const side_pk_p = has_pk ? verified_side(side.pk_wrt_p1(), range_p, range_p, 0, 2) : 0;
bool const right_to_left = side_qk_p1 == 1;

if (side_pk_p == side_qi_p1)
{
int const side_pk_q1 = has_pk ? side.pk_wrt_q1() : 0;

if (side_pk_q1 == 0)
{
ti.operations[0].operation = operation_blocked;
ti.operations[1].operation = right_to_left
? operation_union : operation_intersection;
return;
}

if (side_pk_q1 == side_qk_p1)
{
uu_else_ii(right_to_left, ti);
ti.touch_only = true;
return;
}
}

if (side_pk_p == side_qk_p1)
{
int const side_pk_q2 = has_pk ? side.pk_wrt_q2() : 0;

if (side_pk_q2 == 0)
{
both(ti, operation_continue);
return;
}
if (side_pk_q2 == side_qk_p1)
{
ui_else_iu(right_to_left, ti);
ti.touch_only = true;
return;
}
}
ui_else_iu(! right_to_left, ti);
return;
}
}
};


template
<
typename TurnInfo
>
struct equal : public base_turn_handler
{
template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename IntersectionInfo,
typename DirInfo,
typename SideCalculator,
typename UmbrellaStrategy
>
static inline void apply(UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
TurnInfo& ti,
IntersectionInfo const& info,
DirInfo const& ,
SideCalculator const& side,
UmbrellaStrategy const& umbrella_strategy)
{
assign_point(ti, method_equal, info, non_opposite_to_index(info));

bool const has_pk = ! range_p.is_last_segment();
bool const has_qk = ! range_q.is_last_segment();

int const side_pk_q2 = has_pk && has_qk ? side.pk_wrt_q2() : 0;
int const side_pk_p = has_pk ? side.pk_wrt_p1() : 0;
int const side_qk_p = has_qk ? side.qk_wrt_p1() : 0;

#if ! defined(BOOST_GEOMETRY_USE_RESCALING)

if (has_pk && has_qk && side_pk_p == side_qk_p)
{
typedef typename select_coordinate_type
<
typename UniqueSubRange1::point_type,
typename UniqueSubRange2::point_type
>::type coordinate_type;

typedef detail::distance_measure<coordinate_type> dm_type;

dm_type const dm_pk_q2
= get_distance_measure(range_q.at(1), range_q.at(2), range_p.at(2));
dm_type const dm_qk_p2
= get_distance_measure(range_p.at(1), range_p.at(2), range_q.at(2));

if (dm_qk_p2.measure != dm_pk_q2.measure)
{
ui_else_iu(dm_qk_p2.measure < dm_pk_q2.measure, ti);
return;
}
}
#endif

if (side_pk_q2 == 0 && side_pk_p == side_qk_p)
{
both_collinear<0, 1>(range_p, range_q, umbrella_strategy, 2, 2, ti);
return;
}


if (! opposite(side_pk_p, side_qk_p))
{
ui_else_iu(side_pk_q2 != -1, ti);
}
else
{
ui_else_iu(side_pk_p != -1, ti);
}
}
};

template
<
typename TurnInfo
>
struct start : public base_turn_handler
{
template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename IntersectionInfo,
typename DirInfo,
typename SideCalculator,
typename UmbrellaStrategy
>
static inline bool apply(UniqueSubRange1 const& ,
UniqueSubRange2 const& ,
TurnInfo& ti,
IntersectionInfo const& info,
DirInfo const& dir_info,
SideCalculator const& side,
UmbrellaStrategy const& )
{
#if defined(BOOST_GEOMETRY_USE_RESCALING)
return false;
#endif

BOOST_GEOMETRY_ASSERT(dir_info.how_a != dir_info.how_b);
BOOST_GEOMETRY_ASSERT(dir_info.how_a == -1 || dir_info.how_b == -1);
BOOST_GEOMETRY_ASSERT(dir_info.how_a == 0 || dir_info.how_b == 0);

if (dir_info.how_b == -1)
{

int const side_qj_p1 = side.qj_wrt_p1();
ui_else_iu(side_qj_p1 == -1, ti);
}
else if (dir_info.how_a == -1)
{
int const side_pj_q1 = side.pj_wrt_q1();
ui_else_iu(side_pj_q1 == 1, ti);
}

assign_point_and_correct(ti, method_start, info, dir_info);
return true;
}

};


template
<
typename TurnInfo,
typename AssignPolicy
>
struct equal_opposite : public base_turn_handler
{
template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename OutputIterator,
typename IntersectionInfo
>
static inline void apply(
UniqueSubRange1 const& ,
UniqueSubRange2 const& ,
TurnInfo tp,
OutputIterator& out,
IntersectionInfo const& intersection_info)
{
if (AssignPolicy::include_opposite)
{
tp.method = method_equal;
for (unsigned int i = 0; i < 2; i++)
{
tp.operations[i].operation = operation_opposite;
}
for (unsigned int i = 0; i < intersection_info.i_info().count; i++)
{
assign_point(tp, method_none, intersection_info.i_info(), i);
*out++ = tp;
}
}
}
};

template
<
typename TurnInfo
>
struct collinear : public base_turn_handler
{

template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename IntersectionInfo,
typename DirInfo,
typename SidePolicy
>
static inline void apply(
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
TurnInfo& ti,
IntersectionInfo const& info,
DirInfo const& dir_info,
SidePolicy const& side)
{
assign_point(ti, method_collinear, info, non_opposite_to_index(info));

int const arrival = dir_info.arrival[0];
BOOST_GEOMETRY_ASSERT(arrival != 0);

bool const has_pk = ! range_p.is_last_segment();
bool const has_qk = ! range_q.is_last_segment();
int const side_p = has_pk ? side.pk_wrt_p1() : 0;
int const side_q = has_qk ? side.qk_wrt_q1() : 0;

int const side_p_or_q = arrival == 1
? side_p
: side_q
;


int const product = arrival * side_p_or_q;

if(product == 0)
{
both(ti, operation_continue);
}
else
{
ui_else_iu(product == 1, ti);
}

ti.operations[0].remaining_distance
= side_p == 0 && has_pk
? distance_measure(ti.point, range_p.at(2))
: distance_measure(ti.point, range_p.at(1));
ti.operations[1].remaining_distance
= side_q == 0 && has_qk
? distance_measure(ti.point, range_q.at(2))
: distance_measure(ti.point, range_q.at(1));
}
};

template
<
typename TurnInfo,
typename AssignPolicy
>
struct collinear_opposite : public base_turn_handler
{
private :


template <unsigned int Index, typename IntersectionInfo>
static inline bool set_tp(int side_rk_r, TurnInfo& tp,
IntersectionInfo const& intersection_info)
{
BOOST_STATIC_ASSERT(Index <= 1);

operation_type blocked = operation_blocked;
switch(side_rk_r)
{
case 1 :
tp.operations[Index].operation = operation_intersection;
break;
case -1 :
tp.operations[Index].operation = operation_union;
break;
case 0 :
if (AssignPolicy::include_opposite)
{
tp.operations[Index].operation = operation_opposite;
blocked = operation_opposite;
}
else
{
return false;
}
break;
}

tp.operations[1 - Index].operation = blocked;

assign_point(tp, method_collinear, intersection_info, 1 - Index);
return true;
}

public:
static inline void empty_transformer(TurnInfo &) {}

template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename OutputIterator,
typename IntersectionInfo,
typename SidePolicy
>
static inline void apply(
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,

TurnInfo const& tp_model,
OutputIterator& out,

IntersectionInfo const& intersection_info,
SidePolicy const& side)
{
apply(range_p, range_q,
tp_model, out, intersection_info, side, empty_transformer);
}

template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename OutputIterator,
typename IntersectionInfo,
typename SidePolicy,
typename TurnTransformer
>
static inline void apply(
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,

TurnInfo const& tp_model,
OutputIterator& out,

IntersectionInfo const& info,
SidePolicy const& side,
TurnTransformer turn_transformer)
{
TurnInfo tp = tp_model;

int const p_arrival = info.d_info().arrival[0];
int const q_arrival = info.d_info().arrival[1];

if ( p_arrival == 1
&& ! range_p.is_last_segment()
&& set_tp<0>(side.pk_wrt_p1(), tp, info.i_info()) )
{
turn_transformer(tp);

*out++ = tp;
}

if ( q_arrival == 1
&& ! range_q.is_last_segment()
&& set_tp<1>(side.qk_wrt_q1(), tp, info.i_info()) )
{
turn_transformer(tp);

*out++ = tp;
}

if (AssignPolicy::include_opposite)
{
if ((q_arrival == -1 && p_arrival == 0)
|| (p_arrival == -1 && q_arrival == 0))
{
for (unsigned int i = 0; i < 2; i++)
{
tp.operations[i].operation = operation_opposite;
}
for (unsigned int i = 0; i < info.i_info().count; i++)
{
assign_point(tp, method_collinear, info.i_info(), i);
*out++ = tp;
}
}
}

}
};


template
<
typename TurnInfo
>
struct crosses : public base_turn_handler
{
template <typename IntersectionInfo, typename DirInfo>
static inline void apply(TurnInfo& ti,
IntersectionInfo const& intersection_info,
DirInfo const& dir_info)
{
assign_point(ti, method_crosses, intersection_info, 0);

int const side_qi_p1 = dir_info.sides.template get<1, 0>();
unsigned int const index = side_qi_p1 == 1 ? 0 : 1;
ti.operations[index].operation = operation_union;
ti.operations[1 - index].operation = operation_intersection;
}
};

struct only_convert : public base_turn_handler
{
template<typename TurnInfo, typename IntersectionInfo>
static inline void apply(TurnInfo& ti, IntersectionInfo const& intersection_info)
{
assign_point(ti, method_none, intersection_info, 0);
ti.operations[0].operation = operation_continue;
ti.operations[1].operation = operation_continue;
}
};


struct assign_null_policy
{
static bool const include_no_turn = false;
static bool const include_degenerate = false;
static bool const include_opposite = false;
static bool const include_start_turn = false;
};

struct assign_policy_only_start_turns
{
static bool const include_no_turn = false;
static bool const include_degenerate = false;
static bool const include_opposite = false;
static bool const include_start_turn = true;
};


template<typename AssignPolicy>
struct get_turn_info
{
template
<
typename UniqueSubRange1,
typename UniqueSubRange2,
typename TurnInfo,
typename UmbrellaStrategy,
typename RobustPolicy,
typename OutputIterator
>
static inline OutputIterator apply(
UniqueSubRange1 const& range_p,
UniqueSubRange2 const& range_q,
TurnInfo const& tp_model,
UmbrellaStrategy const& umbrella_strategy,
RobustPolicy const& robust_policy,
OutputIterator out)
{
typedef intersection_info
<
UniqueSubRange1, UniqueSubRange2,
typename TurnInfo::point_type,
UmbrellaStrategy,
RobustPolicy
> inters_info;

inters_info inters(range_p, range_q, umbrella_strategy, robust_policy);

char const method = inters.d_info().how;

if (method == 'd')
{
return out;
}

TurnInfo tp = tp_model;

bool const handle_as_touch_interior = method == 'm';
bool const handle_as_cross = method == 'i';
bool handle_as_touch = method == 't';
bool handle_as_equal = method == 'e';
bool const handle_as_collinear = method == 'c';
bool const handle_as_degenerate = method == '0';
bool const handle_as_start = method == 's';

bool do_only_convert = method == 'a' || method == 'f';

if (handle_as_start)
{
if (AssignPolicy::include_start_turn
&& start<TurnInfo>::apply(range_p, range_q, tp, inters.i_info(), inters.d_info(), inters.sides(), umbrella_strategy))
{
*out++ = tp;
}
else
{
do_only_convert = true;
}
}

if (handle_as_touch_interior)
{
typedef touch_interior<TurnInfo> handler;

if ( inters.d_info().arrival[1] == 1 )
{
if (handler::handle_as_touch(inters.i_info(), range_p))
{
handle_as_touch = true;
}
else
{
handler::template apply<0>(range_p, range_q, tp, inters.i_info(), inters.d_info(),
inters.sides(), umbrella_strategy);
*out++ = tp;
}
}
else
{
if (handler::handle_as_touch(inters.i_info(), range_q))
{
handle_as_touch = true;
}
else
{
handler::template apply<1>(range_q, range_p, tp, inters.i_info(), inters.d_info(),
inters.get_swapped_sides(), umbrella_strategy);
*out++ = tp;
}
}
}

if (handle_as_cross)
{
crosses<TurnInfo>::apply(tp, inters.i_info(), inters.d_info());
*out++ = tp;
}

if (handle_as_touch)
{
touch<TurnInfo>::apply(range_p, range_q, tp, inters.i_info(), inters.d_info(), inters.sides(), umbrella_strategy);
*out++ = tp;
}

if (handle_as_collinear)
{
if ( ! inters.d_info().opposite )
{

if ( inters.d_info().arrival[0] == 0 )
{
handle_as_equal = true;
}
else
{
collinear<TurnInfo>::apply(range_p, range_q, tp,
inters.i_info(), inters.d_info(), inters.sides());
*out++ = tp;
}
}
else
{
collinear_opposite
<
TurnInfo,
AssignPolicy
>::apply(range_p, range_q, tp, out, inters, inters.sides());
}
}

if (handle_as_equal)
{
if ( ! inters.d_info().opposite )
{
equal<TurnInfo>::apply(range_p, range_q, tp,
inters.i_info(), inters.d_info(), inters.sides(),
umbrella_strategy);
if (handle_as_collinear)
{
tp.method = method_collinear;
}
*out++ = tp;
}
else
{
equal_opposite
<
TurnInfo,
AssignPolicy
>::apply(range_p, range_q, tp, out, inters);
}
}

if ((handle_as_degenerate && AssignPolicy::include_degenerate)
|| (do_only_convert && AssignPolicy::include_no_turn))
{
if (inters.i_info().count > 0)
{
only_convert::apply(tp, inters.i_info());
*out++ = tp;
}
}

return out;
}
};


}} 
#endif 


}} 


#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif 
