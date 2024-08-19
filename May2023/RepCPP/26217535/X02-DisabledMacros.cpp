





#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

#include <iostream>

struct foo {
foo(){
REQUIRE_NOTHROW( print() );
}
void print() const {
std::cout << "This should not happen\n";
}
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif
static foo f;


#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-function"
#endif

TEST_CASE( "Disabled Macros" ) {
CHECK( 1 == 2 );
REQUIRE( 1 == 2 );
std::cout << "This should not happen\n";
FAIL();

STATIC_CHECK( 0 == 1 );
STATIC_REQUIRE( !true );

CAPTURE( 1 );
CAPTURE( 1, "captured" );

REQUIRE_THAT( 1,
Catch::Matchers::Predicate( []( int ) { return false; } ) );
BENCHMARK( "Disabled benchmark" ) { REQUIRE( 1 == 2 ); };
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
