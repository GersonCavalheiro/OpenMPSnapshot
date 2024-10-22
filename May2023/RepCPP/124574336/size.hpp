
#ifndef BOOST_RANGE_SIZE_HPP
#define BOOST_RANGE_SIZE_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size_type.hpp>
#include <boost/range/detail/has_member_size.hpp>
#include <boost/assert.hpp>
#include <boost/cstdint.hpp>
#include <boost/utility.hpp>

namespace boost
{
namespace range_detail
{

template<class SinglePassRange>
inline typename ::boost::enable_if<
has_member_size<SinglePassRange>,
typename range_size<const SinglePassRange>::type
>::type
range_calculate_size(const SinglePassRange& rng)
{
return rng.size();
}

template<class SinglePassRange>
inline typename disable_if<
has_member_size<SinglePassRange>,
typename range_size<const SinglePassRange>::type
>::type
range_calculate_size(const SinglePassRange& rng)
{
return std::distance(boost::begin(rng), boost::end(rng));
}
}

template<class SinglePassRange>
inline typename range_size<const SinglePassRange>::type
size(const SinglePassRange& rng)
{
#if BOOST_RANGE_ENABLE_CONCEPT_ASSERT == 1
BOOST_RANGE_CONCEPT_ASSERT((boost::SinglePassRangeConcept<SinglePassRange>));
#endif

#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564)) && \
!BOOST_WORKAROUND(__GNUC__, < 3) \

using namespace range_detail;
#endif

return range_calculate_size(rng);
}

} 

#endif
