


#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generator_exception.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <cstring>


TEST_CASE("Generators -- simple", "[generators]") {
auto i = GENERATE(1, 2, 3);
SECTION("one") {
auto j = GENERATE(values({ -3, -2, -1 }));
REQUIRE(j < i);
}

SECTION("two") {
auto str = GENERATE(as<std::string>{}, "a", "bb", "ccc");
REQUIRE(4u * i > str.size());
}
}

TEST_CASE("3x3x3 ints", "[generators]") {
auto x = GENERATE(1, 2, 3);
auto y = GENERATE(4, 5, 6);
auto z = GENERATE(7, 8, 9);
CHECK(x < y);
CHECK(y < z);
REQUIRE(x < z);
}

TEST_CASE("tables", "[generators]") {

using tuple_type = std::tuple<char const*, int>;
auto data = GENERATE(table<char const*, int>({
tuple_type{"first", 5},
tuple_type{"second", 6},
tuple_type{"third", 5},
tuple_type{"etc...", 6}
}));

REQUIRE(strlen(std::get<0>(data)) == static_cast<size_t>(std::get<1>(data)));
}


#ifdef __cpp_structured_bindings

TEST_CASE( "strlen2", "[approvals][generators]" ) {
using tuple_type = std::tuple<std::string, int>; 
auto [test_input, expected] =
GENERATE( table<std::string, size_t>( { tuple_type{ "one", 3 },
tuple_type{ "two", 3 },
tuple_type{ "three", 5 },
tuple_type{ "four", 4 } } ) );

REQUIRE( test_input.size() == expected );
}
#endif


struct Data { std::string str; size_t len; };

TEST_CASE( "strlen3", "[generators]" ) {
auto data = GENERATE( values<Data>({
{"one", 3},
{"two", 3},
{"three", 5},
{"four", 4}
}));

REQUIRE( data.str.size() == data.len );
}



#ifdef __cpp_structured_bindings



static auto eatCucumbers( int start, int eat ) -> int { return start-eat; }

SCENARIO("Eating cucumbers", "[generators][approvals]") {
using tuple_type = std::tuple<int, int, int>;
auto [start, eat, left] = GENERATE( table<int, int, int>(
{ tuple_type{ 12, 5, 7 }, tuple_type{ 20, 5, 15 } } ) );

GIVEN( "there are " << start << " cucumbers" )
WHEN( "I eat " << eat << " cucumbers" )
THEN( "I should have " << left << " cucumbers" ) {
REQUIRE( eatCucumbers( start, eat ) == left );
}
}
#endif

TEST_CASE("Generators -- adapters", "[generators][generic]") {
SECTION("Filtering by predicate") {
SECTION("Basic usage") {
auto i = GENERATE(filter([] (int val) { return val % 2 == 0; }, values({ 1, 2, 3, 4, 5, 6 })));
REQUIRE(i % 2 == 0);
}
SECTION("Throws if there are no matching values") {
using namespace Catch::Generators;
REQUIRE_THROWS_AS(filter([] (int) {return false; }, value(1)), Catch::GeneratorException);
}
}
SECTION("Shortening a range") {
auto i = GENERATE(take(3, values({ 1, 2, 3, 4, 5, 6 })));
REQUIRE(i < 4);
}
SECTION("Transforming elements") {
SECTION("Same type") {
auto i = GENERATE(map([] (int val) { return val * 2; }, values({ 1, 2, 3 })));
REQUIRE(i % 2 == 0);
}
SECTION("Different type") {
auto i = GENERATE(map<std::string>([] (int val) { return std::to_string(val); }, values({ 1, 2, 3 })));
REQUIRE(i.size() == 1);
}
SECTION("Different deduced type") {
auto i = GENERATE(map([] (int val) { return std::to_string(val); }, values({ 1, 2, 3 })));
REQUIRE(i.size() == 1);
}
}
SECTION("Repeating a generator") {
auto j = GENERATE(repeat(2, values({ 1, 2, 3 })));
REQUIRE(j > 0);
}
SECTION("Chunking a generator into sized pieces") {
SECTION("Number of elements in source is divisible by chunk size") {
auto chunk2 = GENERATE(chunk(2, values({ 1, 1, 2, 2, 3, 3 })));
REQUIRE(chunk2.size() == 2);
REQUIRE(chunk2.front() == chunk2.back());
}
SECTION("Number of elements in source is not divisible by chunk size") {
auto chunk2 = GENERATE(chunk(2, values({ 1, 1, 2, 2, 3 })));
REQUIRE(chunk2.size() == 2);
REQUIRE(chunk2.front() == chunk2.back());
REQUIRE(chunk2.front() < 3);
}
SECTION("Chunk size of zero") {
auto chunk2 = GENERATE(take(3, chunk(0, value(1))));
REQUIRE(chunk2.size() == 0);
}
SECTION("Throws on too small generators") {
using namespace Catch::Generators;
REQUIRE_THROWS_AS(chunk(2, value(1)), Catch::GeneratorException);
}
}
}

TEST_CASE("Random generator", "[generators][approvals]") {
SECTION("Infer int from integral arguments") {
auto val = GENERATE(take(4, random(0, 1)));
STATIC_REQUIRE(std::is_same<decltype(val), int>::value);
REQUIRE(0 <= val);
REQUIRE(val <= 1);
}
SECTION("Infer double from double arguments") {
auto val = GENERATE(take(4, random(0., 1.)));
STATIC_REQUIRE(std::is_same<decltype(val), double>::value);
REQUIRE(0. <= val);
REQUIRE(val < 1);
}
}


TEST_CASE("Nested generators and captured variables", "[generators]") {
using record = std::tuple<int, int>;
auto extent = GENERATE(table<int, int>({
record{3, 7},
record{-5, -3},
record{90, 100}
}));

auto from = std::get<0>(extent);
auto to = std::get<1>(extent);

auto values = GENERATE_COPY(range(from, to));
REQUIRE(values > -6);
}

namespace {
size_t call_count = 0;
size_t test_count = 0;
std::vector<int> make_data() {
return { 1, 3, 5, 7, 9, 11 };
}
std::vector<int> make_data_counted() {
++call_count;
return make_data();
}
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

TEST_CASE("Copy and then generate a range", "[generators]") {
SECTION("from var and iterators") {
static auto data = make_data();

auto elem = GENERATE_REF(from_range(data.begin(), data.end()));
REQUIRE(elem % 2 == 1);
}
SECTION("From a temporary container") {
auto elem = GENERATE(from_range(make_data_counted()));
++test_count;
REQUIRE(elem % 2 == 1);
}
SECTION("Final validation") {
REQUIRE(call_count == 1);
REQUIRE(make_data().size() == test_count);
}
}

TEST_CASE("#1913 - GENERATE inside a for loop should not keep recreating the generator", "[regression][generators]") {
static int counter = 0;
for (int i = 0; i < 3; ++i) {
int _ = GENERATE(1, 2);
(void)_;
++counter;
}
REQUIRE(counter < 7);
}

TEST_CASE("#1913 - GENERATEs can share a line", "[regression][generators]") {
int i = GENERATE(1, 2); int j = GENERATE(3, 4);
REQUIRE(i != j);
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
