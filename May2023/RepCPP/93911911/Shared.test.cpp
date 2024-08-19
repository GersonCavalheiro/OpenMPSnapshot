
#include <catch.hpp>

#ifdef PP_USE_OMP
#include <omp.h>
#endif

#define private public
#include "Shared.h"
#undef private

TEST_CASE( "NUMA-aware initialization without separate workers.", "[numaInitNoSep]" ) {

float l_arr1[73*31];
for( unsigned int l_en = 0; l_en < 73*31; l_en++ ) l_arr1[l_en] = float(1);
float l_arr2[3] = {1, 1, 1};

#ifdef PP_USE_OMP
#pragma omp parallel
#endif
{
edge::parallel::Shared l_shared;
l_shared.init( false );

l_shared.numaInit( 73*31, l_arr1 );
l_shared.numaInit( 3,     l_arr2 );
}

for( unsigned int l_en = 0; l_en < 73*31; l_en++ )  REQUIRE( l_arr1[l_en] == float(0) );
for( unsigned int l_en = 0; l_en <     3; l_en++ )  REQUIRE( l_arr2[l_en] == float(0) );
}

TEST_CASE( "NUMA-aware initialization with separate workers.", "[numaInitSep]" ) {

float l_arr1[73*31];
for( unsigned int l_en = 0; l_en < 73*31; l_en++ ) l_arr1[l_en] = float(1);
float l_arr2[3] = {1, 1, 1};

#ifdef PP_USE_OMP
#pragma omp parallel
#endif
{
edge::parallel::Shared l_shared;
l_shared.init( true );

l_shared.numaInit( 73*31, l_arr1 );
l_shared.numaInit( 3,     l_arr2 );
}

for( unsigned int l_en = 0; l_en < 73*31; l_en++ )  REQUIRE( l_arr1[l_en] == float(0) );
for( unsigned int l_en = 0; l_en <     3; l_en++ )  REQUIRE( l_arr2[l_en] == float(0) );
}