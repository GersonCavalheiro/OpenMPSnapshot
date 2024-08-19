

#ifndef LBT_MATERIAL_STATES_TABLE
#define LBT_MATERIAL_STATES_TABLE
#pragma once

#include <array>
#include <limits>
#include <tuple>

#include "../../constexpr_math/constexpr_math.hpp"
#include "../../unit/units.hpp"
#include "state.hpp"


namespace lbt {
namespace material {

template <std::size_t N>
using States = std::array<State, N>;


template <std::size_t N>
class StatesTable {
public:

template <typename Z, typename X, typename Y>
static constexpr Z equationOfState(X const x, Y const y) noexcept = delete;


protected:

template <typename Z, typename X, typename Y>
static constexpr Z interpolate(X const x, Y const y, States<N> const& states) noexcept {

struct InterpolationPoint {
public:
explicit constexpr InterpolationPoint(X const x_, Y const y_, Z const z_, long double const distance_) noexcept
: x{x_}, y{y_}, z{z_}, distance{distance_} {
return;
}
InterpolationPoint() = delete;
InterpolationPoint(InterpolationPoint const&) = default;
InterpolationPoint& operator= (InterpolationPoint const&) = default;
InterpolationPoint(InterpolationPoint&&) = default;
InterpolationPoint& operator= (InterpolationPoint&&) = default;

X x;
Y y;
Z z;
long double distance; 
};

constexpr X x_nan {std::numeric_limits<long double>::quiet_NaN()};
constexpr Y y_nan {std::numeric_limits<long double>::quiet_NaN()};
constexpr Z z_nan {std::numeric_limits<long double>::quiet_NaN()};
constexpr long double inf {std::numeric_limits<long double>::infinity()};
InterpolationPoint xn_yn {x_nan, y_nan, z_nan, inf};
InterpolationPoint xn_yp {x_nan, y_nan, z_nan, inf};
InterpolationPoint xp_yn {x_nan, y_nan, z_nan, inf};
InterpolationPoint xp_yp {x_nan, y_nan, z_nan, inf};

for (auto const& state: states) {
auto const xs {state.template get<X>()};
auto const ys {state.template get<Y>()};
auto const zs {state.template get<Z>()};
X const dx {x - xs};
Y const dy {y - ys};
long double const distance {lbt::cem::sqrt(lbt::cem::ipow(dx.get(), 2) + lbt::cem::ipow(dy.get(), 2))};
if (lbt::cem::isAlmostEqualEpsRel(x.get(), xs.get()) && lbt::cem::isAlmostEqualEpsRel(y.get(), ys.get())) {
return zs;
} else if ((dx.get() > 0.0L) && (dy.get() > 0.0L) && (distance < xn_yn.distance)) {
xn_yn = InterpolationPoint(xs, ys, zs, distance);
} else if ((dx.get() > 0.0L) && (dy.get() < 0.0L) && (distance < xn_yp.distance)) {
xn_yp = InterpolationPoint(xs, ys, zs, distance);
} else if ((dx.get() < 0.0L) && (dy.get() > 0.0L) && (distance < xp_yn.distance)) {
xp_yn = InterpolationPoint(xs, ys, zs, distance);
} else if ((dx.get() < 0.0L) && (dy.get() < 0.0L) && (distance < xp_yp.distance)) {
xp_yp = InterpolationPoint(xs, ys, zs, distance);
}
}
long double const a {(-xn_yn.x + xp_yn.x).get()};
long double const b {(-xn_yn.x + xn_yp.x).get()};
long double const c {(xn_yn.x - xn_yp.x - xp_yn.x + xp_yp.x).get()};
long double const d {(x - xn_yn.x).get()};
long double const e {(-xn_yn.y + xp_yn.y).get()};
long double const f {(-xn_yn.y + xn_yp.y).get()};
long double const g {(xn_yn.y - xn_yp.y - xp_yn.y + xp_yp.y).get()};
long double const h {(y - xn_yn.y).get()};
long double alpha {(x - xn_yn.x).get()/a};
long double beta {(y - xn_yn.y).get()/f};
if (!lbt::cem::isAlmostEqualEpsRel(c*e, - a*g) && !lbt::cem::isAlmostEqualEpsRel(c*f, - b*g)) {
long double const i {lbt::cem::sqrt(-4.0L*(c*e - a*g)*(d*f - b*h) + lbt::cem::ipow(b*e - a*f + d*g - c*h, 2))};
alpha = -(b*e - a*f + d*g - c*h + i)/(2.0L*c*e - 2.0L*a*g);
beta = (b*e - a*f - d*g + c*h + i)/(2.0L*c*f - 2.0L*b*g);
if ((alpha < 0.0L) || (alpha > 1.0L) || (beta < 0.0L) || (beta > 1.0L)) {
alpha = (-b*e + a*f - d*g + c*h + i)/(2.0L*c*e - 2.0L*a*g);
beta = -(-b*e + a*f + d*g - c*h + i)/(2.0L*c*f - 2.0L*b*g);
}
}
Z const z {(1.0L - alpha)*((1.0L - beta)*xn_yn.z + beta*xn_yp.z) + alpha*((1.0L - beta)*xp_yn.z + beta*xp_yp.z)};
return z;
}

StatesTable() = default;
StatesTable(StatesTable const&) = default;
StatesTable& operator= (StatesTable const&) = default;
StatesTable(StatesTable&&) = default;
StatesTable& operator= (StatesTable&&) = default;

};

}
}

#endif 
