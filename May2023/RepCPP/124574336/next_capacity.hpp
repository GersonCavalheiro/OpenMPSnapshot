#ifndef BOOST_CONTAINER_DETAIL_NEXT_CAPACITY_HPP
#define BOOST_CONTAINER_DETAIL_NEXT_CAPACITY_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/container/detail/config_begin.hpp>
#include <boost/container/detail/workaround.hpp>

#include <boost/container/throw_exception.hpp>
#include <boost/container/detail/min_max.hpp>

#include <boost/static_assert.hpp>

namespace boost {
namespace container {
namespace dtl {

template<unsigned Minimum, unsigned Numerator, unsigned Denominator>
struct grow_factor_ratio
{
BOOST_STATIC_ASSERT(Numerator > Denominator);
BOOST_STATIC_ASSERT(Numerator   < 100);
BOOST_STATIC_ASSERT(Denominator < 100);
BOOST_STATIC_ASSERT(Denominator == 1 || (0 != Numerator % Denominator));

template<class SizeType>
SizeType operator()(const SizeType cur_cap, const SizeType add_min_cap, const SizeType max_cap) const
{
const SizeType overflow_limit  = ((SizeType)-1) / Numerator;

SizeType new_cap = 0;

if(cur_cap <= overflow_limit){
new_cap = cur_cap * Numerator / Denominator;
}
else if(Denominator == 1 || (SizeType(new_cap = cur_cap) / Denominator) > overflow_limit){
new_cap = (SizeType)-1;
}
else{
new_cap *= Numerator;
}
return max_value<SizeType>
( SizeType(Minimum)
, max_value<SizeType>
( SizeType(cur_cap+add_min_cap)
, min_value<SizeType>(max_cap, new_cap))
);
}
};

}  

struct growth_factor_50
: dtl::grow_factor_ratio<0, 3, 2>
{};

struct growth_factor_60
: dtl::grow_factor_ratio<0, 8, 5>
{};

struct growth_factor_100
: dtl::grow_factor_ratio<0, 2, 1>
{};

template<class SizeType>
BOOST_CONTAINER_FORCEINLINE void clamp_by_stored_size_type(SizeType &, SizeType)
{}

template<class SizeType, class SomeStoredSizeType>
BOOST_CONTAINER_FORCEINLINE void clamp_by_stored_size_type(SizeType &s, SomeStoredSizeType)
{
if (s >= SomeStoredSizeType(-1) ) 
s = SomeStoredSizeType(-1);
}

}  
}  

#include <boost/container/detail/config_end.hpp>

#endif   
