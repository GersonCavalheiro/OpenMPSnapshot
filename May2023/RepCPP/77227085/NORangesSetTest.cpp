

#include "adt/NORangesSet.h" 
#include "adt/Range.h"       
#include "adt/RangeTest.h"   
#include <string>            
#include <tuple>             
#include <gtest/gtest.h>     


using rawspeed::NORangesSet;
using rawspeed::Range;

namespace rawspeed_test {

TEST_P(TwoRangesTest, NORangesSetDataSelfTest) {
{
NORangesSet<Range<int>> s;

auto res = s.insert(r0);
ASSERT_TRUE(res);

res = s.insert(r0);
ASSERT_FALSE(res);
}
{
NORangesSet<Range<int>> s;

auto res = s.insert(r1);
ASSERT_TRUE(res);

res = s.insert(r1);
ASSERT_FALSE(res);
}
}

TEST_P(TwoRangesTest, NORangesSetDataTest) {
{
NORangesSet<Range<int>> s;
auto res = s.insert(r0);
ASSERT_TRUE(res);

res = s.insert(r1);
if (AllOverlapped.find(GetParam()) != AllOverlapped.end()) {
ASSERT_FALSE(res);
} else {
ASSERT_TRUE(res);
}
}
{
NORangesSet<Range<int>> s;
auto res = s.insert(r1);
ASSERT_TRUE(res);

res = s.insert(r0);
if (AllOverlapped.find(GetParam()) != AllOverlapped.end()) {
ASSERT_FALSE(res);
} else {
ASSERT_TRUE(res);
}
}
}

using threeRangesType = std::tuple<int, unsigned, int, unsigned, int, unsigned>;
class ThreeRangesTest : public ::testing::TestWithParam<threeRangesType> {
protected:
ThreeRangesTest() = default;
virtual void SetUp() {
r0 = Range<int>(std::get<0>(GetParam()), std::get<1>(GetParam()));
r1 = Range<int>(std::get<2>(GetParam()), std::get<3>(GetParam()));
r2 = Range<int>(std::get<4>(GetParam()), std::get<5>(GetParam()));
}

Range<int> r0;
Range<int> r1;
Range<int> r2;
};
INSTANTIATE_TEST_CASE_P(
Unsigned, ThreeRangesTest,
testing::Combine(testing::Range(0, 3), testing::Range(0U, 3U),
testing::Range(0, 3), testing::Range(0U, 3U),
testing::Range(0, 3), testing::Range(0U, 3U)));

TEST_P(ThreeRangesTest, NORangesSetDataTest) {
NORangesSet<Range<int>> s;
auto res = s.insert(r0);
ASSERT_TRUE(res);

res = s.insert(r1);
ASSERT_EQ(res, !RangesOverlap(r1, r0));
if (!res)
return; 

res = s.insert(r2);
ASSERT_EQ(res, !RangesOverlap(r0, r2) && !RangesOverlap(r1, r2));
}

} 
