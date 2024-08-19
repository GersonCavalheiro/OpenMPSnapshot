

#include "adt/Point.h"   
#include <algorithm>     
#include <iterator>      
#include <limits>        
#include <ostream>       
#include <string>        
#include <tuple>         
#include <type_traits>   
#include <utility>       
#include <variant>       
#include <gtest/gtest.h> 


using rawspeed::iPoint2D;
using std::make_pair;
using std::move;
using std::numeric_limits;
using std::pair;
using std::tuple;

namespace rawspeed {

::std::ostream& operator<<(::std::ostream& os, const iPoint2D p) {
return os << "(" << p.x << ", " << p.y << ")";
}

} 

namespace rawspeed_test {

static constexpr iPoint2D::area_type maxVal =
numeric_limits<iPoint2D::value_type>::max();
static constexpr iPoint2D::area_type minVal =
numeric_limits<iPoint2D::value_type>::min();
static constexpr iPoint2D::area_type absMinVal = -minVal;
static constexpr iPoint2D::area_type maxAreaVal = maxVal * maxVal;
static constexpr iPoint2D::area_type minAreaVal = absMinVal * absMinVal;
static constexpr iPoint2D::area_type mixAreaVal = maxVal * absMinVal;

TEST(PointTest, Constructor) {
int x = -10, y = 15;
ASSERT_NO_THROW({
iPoint2D a;
ASSERT_EQ(a.x, 0);
ASSERT_EQ(a.y, 0);
});
ASSERT_NO_THROW({
iPoint2D a(x, y);
ASSERT_EQ(a.x, x);
ASSERT_EQ(a.y, y);
});
ASSERT_NO_THROW({
const iPoint2D a(x, y);
iPoint2D b(a);
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
iPoint2D a(x, y);
iPoint2D b(a);
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
const iPoint2D a(x, y);
iPoint2D b(std::move(a));
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
iPoint2D a(x, y);
iPoint2D b(std::move(a));
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
}

TEST(PointTest, AssignmentConstructor) {
int x = -10, y = 15;
ASSERT_NO_THROW({
iPoint2D a(x, y);
iPoint2D b(666, 777);
b = a;
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
const iPoint2D a(x, y);
iPoint2D b(666, 777);
b = a;
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
iPoint2D a(x, y);
iPoint2D b(666, 777);
b = std::move(a);
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
ASSERT_NO_THROW({
const iPoint2D a(x, y);
iPoint2D b(666, 777);
b = std::move(a);
ASSERT_EQ(b.x, x);
ASSERT_EQ(b.y, y);
});
}

TEST(PointTest, EqualityOperator) {
ASSERT_NO_THROW({
const iPoint2D a(18, -12);
const iPoint2D b(18, -12);
ASSERT_EQ(a, b);
ASSERT_EQ(b, a);
});
}

TEST(PointTest, NonEqualityOperator) {
ASSERT_NO_THROW({
const iPoint2D a(777, 888);
const iPoint2D b(888, 777);
const iPoint2D c(128, 256);
ASSERT_NE(a, b);
ASSERT_NE(b, a);
ASSERT_NE(a, c);
ASSERT_NE(c, a);
ASSERT_NE(b, c);
ASSERT_NE(c, b);
});
}

using IntPair = pair<int, int>;
using Six = std::tuple<IntPair, IntPair, IntPair>;
class PointTest : public ::testing::TestWithParam<Six> {
protected:
PointTest() = default;
virtual void SetUp() {
auto p = GetParam();

auto pair = std::get<0>(p);
a = iPoint2D(pair.first, pair.second);

pair = std::get<1>(p);
b = iPoint2D(pair.first, pair.second);

pair = std::get<2>(p);
c = iPoint2D(pair.first, pair.second);
}

iPoint2D a;
iPoint2D b;
iPoint2D c;
};


static const Six valueSum[]{
make_tuple(make_pair(-5, -5), make_pair(-5, -5), make_pair(-10, -10)),
make_tuple(make_pair(-5, -5), make_pair(-5, 0), make_pair(-10, -5)),
make_tuple(make_pair(-5, -5), make_pair(-5, 5), make_pair(-10, 0)),
make_tuple(make_pair(-5, -5), make_pair(0, -5), make_pair(-5, -10)),
make_tuple(make_pair(-5, -5), make_pair(0, 0), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(0, 5), make_pair(-5, 0)),
make_tuple(make_pair(-5, -5), make_pair(5, -5), make_pair(0, -10)),
make_tuple(make_pair(-5, -5), make_pair(5, 0), make_pair(0, -5)),
make_tuple(make_pair(-5, -5), make_pair(5, 5), make_pair(0, 0)),
make_tuple(make_pair(-5, 0), make_pair(-5, -5), make_pair(-10, -5)),
make_tuple(make_pair(-5, 0), make_pair(-5, 0), make_pair(-10, 0)),
make_tuple(make_pair(-5, 0), make_pair(-5, 5), make_pair(-10, 5)),
make_tuple(make_pair(-5, 0), make_pair(0, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 0), make_pair(0, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(0, 5), make_pair(-5, 5)),
make_tuple(make_pair(-5, 0), make_pair(5, -5), make_pair(0, -5)),
make_tuple(make_pair(-5, 0), make_pair(5, 0), make_pair(0, 0)),
make_tuple(make_pair(-5, 0), make_pair(5, 5), make_pair(0, 5)),
make_tuple(make_pair(-5, 5), make_pair(-5, -5), make_pair(-10, 0)),
make_tuple(make_pair(-5, 5), make_pair(-5, 0), make_pair(-10, 5)),
make_tuple(make_pair(-5, 5), make_pair(-5, 5), make_pair(-10, 10)),
make_tuple(make_pair(-5, 5), make_pair(0, -5), make_pair(-5, 0)),
make_tuple(make_pair(-5, 5), make_pair(0, 0), make_pair(-5, 5)),
make_tuple(make_pair(-5, 5), make_pair(0, 5), make_pair(-5, 10)),
make_tuple(make_pair(-5, 5), make_pair(5, -5), make_pair(0, 0)),
make_tuple(make_pair(-5, 5), make_pair(5, 0), make_pair(0, 5)),
make_tuple(make_pair(-5, 5), make_pair(5, 5), make_pair(0, 10)),
make_tuple(make_pair(0, -5), make_pair(-5, -5), make_pair(-5, -10)),
make_tuple(make_pair(0, -5), make_pair(-5, 0), make_pair(-5, -5)),
make_tuple(make_pair(0, -5), make_pair(-5, 5), make_pair(-5, 0)),
make_tuple(make_pair(0, -5), make_pair(0, -5), make_pair(0, -10)),
make_tuple(make_pair(0, -5), make_pair(0, 0), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(0, 5), make_pair(0, 0)),
make_tuple(make_pair(0, -5), make_pair(5, -5), make_pair(5, -10)),
make_tuple(make_pair(0, -5), make_pair(5, 0), make_pair(5, -5)),
make_tuple(make_pair(0, -5), make_pair(5, 5), make_pair(5, 0)),
make_tuple(make_pair(0, 0), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(0, 0), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(0, 0), make_pair(-5, 5), make_pair(-5, 5)),
make_tuple(make_pair(0, 0), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(0, 0), make_pair(0, 0), make_pair(0, 0)),
make_tuple(make_pair(0, 0), make_pair(0, 5), make_pair(0, 5)),
make_tuple(make_pair(0, 0), make_pair(5, -5), make_pair(5, -5)),
make_tuple(make_pair(0, 0), make_pair(5, 0), make_pair(5, 0)),
make_tuple(make_pair(0, 0), make_pair(5, 5), make_pair(5, 5)),
make_tuple(make_pair(0, 5), make_pair(-5, -5), make_pair(-5, 0)),
make_tuple(make_pair(0, 5), make_pair(-5, 0), make_pair(-5, 5)),
make_tuple(make_pair(0, 5), make_pair(-5, 5), make_pair(-5, 10)),
make_tuple(make_pair(0, 5), make_pair(0, -5), make_pair(0, 0)),
make_tuple(make_pair(0, 5), make_pair(0, 0), make_pair(0, 5)),
make_tuple(make_pair(0, 5), make_pair(0, 5), make_pair(0, 10)),
make_tuple(make_pair(0, 5), make_pair(5, -5), make_pair(5, 0)),
make_tuple(make_pair(0, 5), make_pair(5, 0), make_pair(5, 5)),
make_tuple(make_pair(0, 5), make_pair(5, 5), make_pair(5, 10)),
make_tuple(make_pair(5, -5), make_pair(-5, -5), make_pair(0, -10)),
make_tuple(make_pair(5, -5), make_pair(-5, 0), make_pair(0, -5)),
make_tuple(make_pair(5, -5), make_pair(-5, 5), make_pair(0, 0)),
make_tuple(make_pair(5, -5), make_pair(0, -5), make_pair(5, -10)),
make_tuple(make_pair(5, -5), make_pair(0, 0), make_pair(5, -5)),
make_tuple(make_pair(5, -5), make_pair(0, 5), make_pair(5, 0)),
make_tuple(make_pair(5, -5), make_pair(5, -5), make_pair(10, -10)),
make_tuple(make_pair(5, -5), make_pair(5, 0), make_pair(10, -5)),
make_tuple(make_pair(5, -5), make_pair(5, 5), make_pair(10, 0)),
make_tuple(make_pair(5, 0), make_pair(-5, -5), make_pair(0, -5)),
make_tuple(make_pair(5, 0), make_pair(-5, 0), make_pair(0, 0)),
make_tuple(make_pair(5, 0), make_pair(-5, 5), make_pair(0, 5)),
make_tuple(make_pair(5, 0), make_pair(0, -5), make_pair(5, -5)),
make_tuple(make_pair(5, 0), make_pair(0, 0), make_pair(5, 0)),
make_tuple(make_pair(5, 0), make_pair(0, 5), make_pair(5, 5)),
make_tuple(make_pair(5, 0), make_pair(5, -5), make_pair(10, -5)),
make_tuple(make_pair(5, 0), make_pair(5, 0), make_pair(10, 0)),
make_tuple(make_pair(5, 0), make_pair(5, 5), make_pair(10, 5)),
make_tuple(make_pair(5, 5), make_pair(-5, -5), make_pair(0, 0)),
make_tuple(make_pair(5, 5), make_pair(-5, 0), make_pair(0, 5)),
make_tuple(make_pair(5, 5), make_pair(-5, 5), make_pair(0, 10)),
make_tuple(make_pair(5, 5), make_pair(0, -5), make_pair(5, 0)),
make_tuple(make_pair(5, 5), make_pair(0, 0), make_pair(5, 5)),
make_tuple(make_pair(5, 5), make_pair(0, 5), make_pair(5, 10)),
make_tuple(make_pair(5, 5), make_pair(5, -5), make_pair(10, 0)),
make_tuple(make_pair(5, 5), make_pair(5, 0), make_pair(10, 5)),
make_tuple(make_pair(5, 5), make_pair(5, 5), make_pair(10, 10)),
};

INSTANTIATE_TEST_CASE_P(SumTest, PointTest, ::testing::ValuesIn(valueSum));
TEST_P(PointTest, InPlaceAddTest1) {
ASSERT_NO_THROW({
a += b;
ASSERT_EQ(a, c);
});
}
TEST_P(PointTest, InPlaceAddTest2) {
ASSERT_NO_THROW({
b += a;
ASSERT_EQ(b, c);
});
}
TEST_P(PointTest, AddTest1) {
ASSERT_NO_THROW({
iPoint2D d = a + b;
ASSERT_EQ(d, c);
});
}
TEST_P(PointTest, AddTest2) {
ASSERT_NO_THROW({
iPoint2D d = b + a;
ASSERT_EQ(d, c);
});
}

TEST_P(PointTest, InPlaceSubTest1) {
ASSERT_NO_THROW({
c -= a;
ASSERT_EQ(c, b);
});
}
TEST_P(PointTest, InPlaceSubTest2) {
ASSERT_NO_THROW({
c -= b;
ASSERT_EQ(c, a);
});
}
TEST_P(PointTest, SubTest1) {
ASSERT_NO_THROW({
iPoint2D d = c - a;
ASSERT_EQ(d, b);
});
}
TEST_P(PointTest, SubTest2) {
ASSERT_NO_THROW({
iPoint2D d = c - b;
ASSERT_EQ(d, a);
});
}

using hasPositiveAreaType = std::tuple<int, int>;
class HasPositiveAreaTest
: public ::testing::TestWithParam<hasPositiveAreaType> {
protected:
HasPositiveAreaTest() = default;
virtual void SetUp() {
auto param = GetParam();
p = {std::get<0>(param), std::get<1>(param)};
}

iPoint2D p;
};
INSTANTIATE_TEST_CASE_P(HasPositiveAreaTest, HasPositiveAreaTest,
::testing::Combine(::testing::Range(-2, 3),
::testing::Range(-2, 3)));
static const iPoint2D PositiveAreaData[] = {
{1, 1},
{1, 2},
{2, 1},
{2, 2},
};
TEST_P(HasPositiveAreaTest, HasPositiveAreaTest) {
ASSERT_NO_THROW({
ASSERT_EQ(p.hasPositiveArea(), std::find(std::cbegin(PositiveAreaData),
std::cend(PositiveAreaData),
p) != std::cend(PositiveAreaData));
});
}

using areaType = tuple<IntPair, iPoint2D::area_type>;
class AreaTest : public ::testing::TestWithParam<areaType> {
protected:
AreaTest() = default;
virtual void SetUp() {
auto param = GetParam();

auto pair = std::get<0>(param);
p = iPoint2D(pair.first, pair.second);

a = std::get<1>(param);
}

iPoint2D p;
iPoint2D::area_type a;
};


static const areaType valueArea[]{
make_tuple(make_pair(-5, -5), 25),
make_tuple(make_pair(-5, 0), 0),
make_tuple(make_pair(-5, 5), 25),
make_tuple(make_pair(0, -5), 0),
make_tuple(make_pair(0, 0), 0),
make_tuple(make_pair(0, 5), 0),
make_tuple(make_pair(5, -5), 25),
make_tuple(make_pair(5, 0), 0),
make_tuple(make_pair(5, 5), 25),

make_tuple(make_pair(minVal, 0), 0),
make_tuple(make_pair(maxVal, 0), 0),
make_tuple(make_pair(minVal, -1), absMinVal),
make_tuple(make_pair(maxVal, -1), maxVal),
make_tuple(make_pair(minVal, 1), absMinVal),
make_tuple(make_pair(maxVal, 1), maxVal),

make_tuple(make_pair(0, minVal), 0),
make_tuple(make_pair(0, maxVal), 0),
make_tuple(make_pair(-1, minVal), absMinVal),
make_tuple(make_pair(-1, maxVal), maxVal),
make_tuple(make_pair(1, minVal), absMinVal),
make_tuple(make_pair(1, maxVal), maxVal),

make_tuple(make_pair(minVal, minVal), minAreaVal),
make_tuple(make_pair(minVal, maxVal), mixAreaVal),
make_tuple(make_pair(maxVal, minVal), mixAreaVal),
make_tuple(make_pair(maxVal, maxVal), maxAreaVal),

};
INSTANTIATE_TEST_CASE_P(AreaTest, AreaTest, ::testing::ValuesIn(valueArea));
TEST_P(AreaTest, AreaTest) {
ASSERT_NO_THROW({ ASSERT_EQ(p.area(), a); });
}

using operatorsType =
std::tuple<IntPair, IntPair, bool, bool, bool, bool, bool>;
class OperatorsTest : public ::testing::TestWithParam<operatorsType> {
protected:
OperatorsTest() = default;
virtual void SetUp() {
auto p = GetParam();

auto pair = std::get<0>(p);
a = iPoint2D(pair.first, pair.second);

pair = std::get<1>(p);
b = iPoint2D(pair.first, pair.second);

eq = std::get<2>(p);
lt = std::get<3>(p);
gt = std::get<4>(p);
le = std::get<5>(p);
ge = std::get<6>(p);
}

iPoint2D a;
iPoint2D b;
bool eq;
bool lt;
bool gt;
bool le;
bool ge;
};


static const operatorsType operatorsValues[]{
make_tuple(make_pair(-1, -1), make_pair(-1, -1), true, false, false, true,
true),
make_tuple(make_pair(-1, -1), make_pair(-1, 0), false, false, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(-1, 1), false, false, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(0, -1), false, false, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(0, 0), false, true, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(0, 1), false, true, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(1, -1), false, false, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(1, 0), false, true, false, true,
false),
make_tuple(make_pair(-1, -1), make_pair(1, 1), false, true, false, true,
false),
make_tuple(make_pair(-1, 0), make_pair(-1, -1), false, false, false, false,
true),
make_tuple(make_pair(-1, 0), make_pair(-1, 0), true, false, false, true,
true),
make_tuple(make_pair(-1, 0), make_pair(-1, 1), false, false, false, true,
false),
make_tuple(make_pair(-1, 0), make_pair(0, -1), false, false, false, false,
false),
make_tuple(make_pair(-1, 0), make_pair(0, 0), false, false, false, true,
false),
make_tuple(make_pair(-1, 0), make_pair(0, 1), false, true, false, true,
false),
make_tuple(make_pair(-1, 0), make_pair(1, -1), false, false, false, false,
false),
make_tuple(make_pair(-1, 0), make_pair(1, 0), false, false, false, true,
false),
make_tuple(make_pair(-1, 0), make_pair(1, 1), false, true, false, true,
false),
make_tuple(make_pair(-1, 1), make_pair(-1, -1), false, false, false, false,
true),
make_tuple(make_pair(-1, 1), make_pair(-1, 0), false, false, false, false,
true),
make_tuple(make_pair(-1, 1), make_pair(-1, 1), true, false, false, true,
true),
make_tuple(make_pair(-1, 1), make_pair(0, -1), false, false, false, false,
false),
make_tuple(make_pair(-1, 1), make_pair(0, 0), false, false, false, false,
false),
make_tuple(make_pair(-1, 1), make_pair(0, 1), false, false, false, true,
false),
make_tuple(make_pair(-1, 1), make_pair(1, -1), false, false, false, false,
false),
make_tuple(make_pair(-1, 1), make_pair(1, 0), false, false, false, false,
false),
make_tuple(make_pair(-1, 1), make_pair(1, 1), false, false, false, true,
false),
make_tuple(make_pair(0, -1), make_pair(-1, -1), false, false, false, false,
true),
make_tuple(make_pair(0, -1), make_pair(-1, 0), false, false, false, false,
false),
make_tuple(make_pair(0, -1), make_pair(-1, 1), false, false, false, false,
false),
make_tuple(make_pair(0, -1), make_pair(0, -1), true, false, false, true,
true),
make_tuple(make_pair(0, -1), make_pair(0, 0), false, false, false, true,
false),
make_tuple(make_pair(0, -1), make_pair(0, 1), false, false, false, true,
false),
make_tuple(make_pair(0, -1), make_pair(1, -1), false, false, false, true,
false),
make_tuple(make_pair(0, -1), make_pair(1, 0), false, true, false, true,
false),
make_tuple(make_pair(0, -1), make_pair(1, 1), false, true, false, true,
false),
make_tuple(make_pair(0, 0), make_pair(-1, -1), false, false, true, false,
true),
make_tuple(make_pair(0, 0), make_pair(-1, 0), false, false, false, false,
true),
make_tuple(make_pair(0, 0), make_pair(-1, 1), false, false, false, false,
false),
make_tuple(make_pair(0, 0), make_pair(0, -1), false, false, false, false,
true),
make_tuple(make_pair(0, 0), make_pair(0, 0), true, false, false, true,
true),
make_tuple(make_pair(0, 0), make_pair(0, 1), false, false, false, true,
false),
make_tuple(make_pair(0, 0), make_pair(1, -1), false, false, false, false,
false),
make_tuple(make_pair(0, 0), make_pair(1, 0), false, false, false, true,
false),
make_tuple(make_pair(0, 0), make_pair(1, 1), false, true, false, true,
false),
make_tuple(make_pair(0, 1), make_pair(-1, -1), false, false, true, false,
true),
make_tuple(make_pair(0, 1), make_pair(-1, 0), false, false, true, false,
true),
make_tuple(make_pair(0, 1), make_pair(-1, 1), false, false, false, false,
true),
make_tuple(make_pair(0, 1), make_pair(0, -1), false, false, false, false,
true),
make_tuple(make_pair(0, 1), make_pair(0, 0), false, false, false, false,
true),
make_tuple(make_pair(0, 1), make_pair(0, 1), true, false, false, true,
true),
make_tuple(make_pair(0, 1), make_pair(1, -1), false, false, false, false,
false),
make_tuple(make_pair(0, 1), make_pair(1, 0), false, false, false, false,
false),
make_tuple(make_pair(0, 1), make_pair(1, 1), false, false, false, true,
false),
make_tuple(make_pair(1, -1), make_pair(-1, -1), false, false, false, false,
true),
make_tuple(make_pair(1, -1), make_pair(-1, 0), false, false, false, false,
false),
make_tuple(make_pair(1, -1), make_pair(-1, 1), false, false, false, false,
false),
make_tuple(make_pair(1, -1), make_pair(0, -1), false, false, false, false,
true),
make_tuple(make_pair(1, -1), make_pair(0, 0), false, false, false, false,
false),
make_tuple(make_pair(1, -1), make_pair(0, 1), false, false, false, false,
false),
make_tuple(make_pair(1, -1), make_pair(1, -1), true, false, false, true,
true),
make_tuple(make_pair(1, -1), make_pair(1, 0), false, false, false, true,
false),
make_tuple(make_pair(1, -1), make_pair(1, 1), false, false, false, true,
false),
make_tuple(make_pair(1, 0), make_pair(-1, -1), false, false, true, false,
true),
make_tuple(make_pair(1, 0), make_pair(-1, 0), false, false, false, false,
true),
make_tuple(make_pair(1, 0), make_pair(-1, 1), false, false, false, false,
false),
make_tuple(make_pair(1, 0), make_pair(0, -1), false, false, true, false,
true),
make_tuple(make_pair(1, 0), make_pair(0, 0), false, false, false, false,
true),
make_tuple(make_pair(1, 0), make_pair(0, 1), false, false, false, false,
false),
make_tuple(make_pair(1, 0), make_pair(1, -1), false, false, false, false,
true),
make_tuple(make_pair(1, 0), make_pair(1, 0), true, false, false, true,
true),
make_tuple(make_pair(1, 0), make_pair(1, 1), false, false, false, true,
false),
make_tuple(make_pair(1, 1), make_pair(-1, -1), false, false, true, false,
true),
make_tuple(make_pair(1, 1), make_pair(-1, 0), false, false, true, false,
true),
make_tuple(make_pair(1, 1), make_pair(-1, 1), false, false, false, false,
true),
make_tuple(make_pair(1, 1), make_pair(0, -1), false, false, true, false,
true),
make_tuple(make_pair(1, 1), make_pair(0, 0), false, false, true, false,
true),
make_tuple(make_pair(1, 1), make_pair(0, 1), false, false, false, false,
true),
make_tuple(make_pair(1, 1), make_pair(1, -1), false, false, false, false,
true),
make_tuple(make_pair(1, 1), make_pair(1, 0), false, false, false, false,
true),
make_tuple(make_pair(1, 1), make_pair(1, 1), true, false, false, true,
true)};

INSTANTIATE_TEST_CASE_P(OperatorsTests, OperatorsTest,
::testing::ValuesIn(operatorsValues));

TEST_P(OperatorsTest, OperatorEQTest) {
ASSERT_NO_THROW({ ASSERT_EQ(a == b, eq); });
ASSERT_NO_THROW({ ASSERT_EQ(b == a, eq); });
}
TEST_P(OperatorsTest, OperatorNETest) {
ASSERT_NO_THROW({ ASSERT_EQ(a != b, !eq); });
ASSERT_NO_THROW({ ASSERT_EQ(b != a, !eq); });
}

TEST_P(OperatorsTest, OperatorLTTest) {
ASSERT_NO_THROW({ ASSERT_EQ(a < b, lt); });
ASSERT_NO_THROW({ ASSERT_EQ(b > a, lt); });
}
TEST_P(OperatorsTest, OperatorGTest) {
ASSERT_NO_THROW({ ASSERT_EQ(a > b, gt); });
ASSERT_NO_THROW({ ASSERT_EQ(b < a, gt); });
}

TEST_P(OperatorsTest, OperatorLETest) {
ASSERT_NO_THROW({ ASSERT_EQ(a <= b, le); });
ASSERT_NO_THROW({ ASSERT_EQ(b >= a, le); });
}
TEST_P(OperatorsTest, OperatorGEest) {
ASSERT_NO_THROW({ ASSERT_EQ(a >= b, ge); });
ASSERT_NO_THROW({ ASSERT_EQ(b <= a, ge); });
}

TEST_P(OperatorsTest, OperatorsTest) {
ASSERT_NO_THROW({ ASSERT_EQ(a.isThisInside(b), le); });
}


static const Six smallestValues[]{
make_tuple(make_pair(-5, -5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(-5, 0), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(-5, 5), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(0, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(0, 0), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(0, 5), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(5, 0), make_pair(-5, -5)),
make_tuple(make_pair(-5, -5), make_pair(5, 5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 0), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 0), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(-5, 5), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(0, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 0), make_pair(0, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(0, 5), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 0), make_pair(5, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 0), make_pair(5, 5), make_pair(-5, 0)),
make_tuple(make_pair(-5, 5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 5), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 5), make_pair(-5, 5), make_pair(-5, 5)),
make_tuple(make_pair(-5, 5), make_pair(0, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 5), make_pair(0, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 5), make_pair(0, 5), make_pair(-5, 5)),
make_tuple(make_pair(-5, 5), make_pair(5, -5), make_pair(-5, -5)),
make_tuple(make_pair(-5, 5), make_pair(5, 0), make_pair(-5, 0)),
make_tuple(make_pair(-5, 5), make_pair(5, 5), make_pair(-5, 5)),
make_tuple(make_pair(0, -5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(0, -5), make_pair(-5, 0), make_pair(-5, -5)),
make_tuple(make_pair(0, -5), make_pair(-5, 5), make_pair(-5, -5)),
make_tuple(make_pair(0, -5), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(0, 0), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(0, 5), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(5, -5), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(5, 0), make_pair(0, -5)),
make_tuple(make_pair(0, -5), make_pair(5, 5), make_pair(0, -5)),
make_tuple(make_pair(0, 0), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(0, 0), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(0, 0), make_pair(-5, 5), make_pair(-5, 0)),
make_tuple(make_pair(0, 0), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(0, 0), make_pair(0, 0), make_pair(0, 0)),
make_tuple(make_pair(0, 0), make_pair(0, 5), make_pair(0, 0)),
make_tuple(make_pair(0, 0), make_pair(5, -5), make_pair(0, -5)),
make_tuple(make_pair(0, 0), make_pair(5, 0), make_pair(0, 0)),
make_tuple(make_pair(0, 0), make_pair(5, 5), make_pair(0, 0)),
make_tuple(make_pair(0, 5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(0, 5), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(0, 5), make_pair(-5, 5), make_pair(-5, 5)),
make_tuple(make_pair(0, 5), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(0, 5), make_pair(0, 0), make_pair(0, 0)),
make_tuple(make_pair(0, 5), make_pair(0, 5), make_pair(0, 5)),
make_tuple(make_pair(0, 5), make_pair(5, -5), make_pair(0, -5)),
make_tuple(make_pair(0, 5), make_pair(5, 0), make_pair(0, 0)),
make_tuple(make_pair(0, 5), make_pair(5, 5), make_pair(0, 5)),
make_tuple(make_pair(5, -5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(5, -5), make_pair(-5, 0), make_pair(-5, -5)),
make_tuple(make_pair(5, -5), make_pair(-5, 5), make_pair(-5, -5)),
make_tuple(make_pair(5, -5), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(5, -5), make_pair(0, 0), make_pair(0, -5)),
make_tuple(make_pair(5, -5), make_pair(0, 5), make_pair(0, -5)),
make_tuple(make_pair(5, -5), make_pair(5, -5), make_pair(5, -5)),
make_tuple(make_pair(5, -5), make_pair(5, 0), make_pair(5, -5)),
make_tuple(make_pair(5, -5), make_pair(5, 5), make_pair(5, -5)),
make_tuple(make_pair(5, 0), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(5, 0), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(5, 0), make_pair(-5, 5), make_pair(-5, 0)),
make_tuple(make_pair(5, 0), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(5, 0), make_pair(0, 0), make_pair(0, 0)),
make_tuple(make_pair(5, 0), make_pair(0, 5), make_pair(0, 0)),
make_tuple(make_pair(5, 0), make_pair(5, -5), make_pair(5, -5)),
make_tuple(make_pair(5, 0), make_pair(5, 0), make_pair(5, 0)),
make_tuple(make_pair(5, 0), make_pair(5, 5), make_pair(5, 0)),
make_tuple(make_pair(5, 5), make_pair(-5, -5), make_pair(-5, -5)),
make_tuple(make_pair(5, 5), make_pair(-5, 0), make_pair(-5, 0)),
make_tuple(make_pair(5, 5), make_pair(-5, 5), make_pair(-5, 5)),
make_tuple(make_pair(5, 5), make_pair(0, -5), make_pair(0, -5)),
make_tuple(make_pair(5, 5), make_pair(0, 0), make_pair(0, 0)),
make_tuple(make_pair(5, 5), make_pair(0, 5), make_pair(0, 5)),
make_tuple(make_pair(5, 5), make_pair(5, -5), make_pair(5, -5)),
make_tuple(make_pair(5, 5), make_pair(5, 0), make_pair(5, 0)),
make_tuple(make_pair(5, 5), make_pair(5, 5), make_pair(5, 5)),
};

class SmallestTest : public PointTest {};

INSTANTIATE_TEST_CASE_P(GetSmallestTest, SmallestTest,
::testing::ValuesIn(smallestValues));
TEST_P(SmallestTest, GetSmallestTest) {
ASSERT_NO_THROW({
ASSERT_EQ(a.getSmallest(b), c);
ASSERT_EQ(a.getSmallest(c), c);
ASSERT_EQ(b.getSmallest(a), c);
ASSERT_EQ(b.getSmallest(c), c);
ASSERT_EQ(c.getSmallest(a), c);
ASSERT_EQ(c.getSmallest(b), c);
ASSERT_EQ(c.getSmallest(c), c);
});
}

} 
