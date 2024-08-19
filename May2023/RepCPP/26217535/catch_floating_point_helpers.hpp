

#ifndef CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED
#define CATCH_FLOATING_POINT_HELPERS_HPP_INCLUDED

#include <catch2/internal/catch_polyfills.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <utility>
#include <limits>

namespace Catch {
namespace Detail {

uint32_t convertToBits(float f);
uint64_t convertToBits(double d);

} 



#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif


template <typename FP>
uint64_t ulpDistance( FP lhs, FP rhs ) {
assert( std::numeric_limits<FP>::is_iec559 &&
"ulpDistance assumes IEEE-754 format for floating point types" );
assert( !Catch::isnan( lhs ) &&
"Distance between NaN and number is not meaningful" );
assert( !Catch::isnan( rhs ) &&
"Distance between NaN and number is not meaningful" );

if ( lhs == rhs ) { return 0; }

static constexpr FP positive_zero{};

if ( lhs == positive_zero ) { lhs = positive_zero; }
if ( rhs == positive_zero ) { rhs = positive_zero; }

if ( std::signbit( lhs ) != std::signbit( rhs ) ) {
return ulpDistance( std::abs( lhs ), positive_zero ) +
ulpDistance( std::abs( rhs ), positive_zero );
}

uint64_t lc = Detail::convertToBits( lhs );
uint64_t rc = Detail::convertToBits( rhs );

if ( lc < rc ) {
std::swap( lc, rc );
}

return lc - rc;
}

#if defined( __GNUC__ ) || defined( __clang__ )
#    pragma GCC diagnostic pop
#endif


} 

#endif 
