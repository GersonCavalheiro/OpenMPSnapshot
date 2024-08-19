#ifndef BOOST_RANGE_COUNTING_RANGE_HPP_INCLUDED
#define BOOST_RANGE_COUNTING_RANGE_HPP_INCLUDED

#include <boost/config.hpp>
#if BOOST_MSVC >= 1400
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <boost/range/iterator_range_core.hpp>
#include <boost/range/value_type.hpp>
#include <boost/range/iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace boost
{
template<class Value>
inline iterator_range<counting_iterator<Value> >
counting_range(Value first, Value last)
{
typedef counting_iterator<Value> counting_iterator_t;
typedef iterator_range<counting_iterator_t> result_t;
return result_t(counting_iterator_t(first),
counting_iterator_t(last));
}

template<class Range>
inline iterator_range<
counting_iterator<
BOOST_DEDUCED_TYPENAME range_iterator<const Range>::type
>
>
counting_range(const Range& rng)
{
typedef counting_iterator<
BOOST_DEDUCED_TYPENAME range_iterator<const Range>::type
> counting_iterator_t;

typedef iterator_range<counting_iterator_t> result_t;

return result_t(counting_iterator_t(boost::begin(rng)),
counting_iterator_t(boost::end(rng)));
}

template<class Range>
inline iterator_range<
counting_iterator<
BOOST_DEDUCED_TYPENAME range_iterator<Range>::type
>
>
counting_range(Range& rng)
{
typedef counting_iterator<
BOOST_DEDUCED_TYPENAME range_iterator<Range>::type
> counting_iterator_t;

typedef iterator_range<counting_iterator_t> result_t;

return result_t(counting_iterator_t(boost::begin(rng)),
counting_iterator_t(boost::end(rng)));
}
} 

#if BOOST_MSVC >= 1400
#pragma warning(pop)
#endif

#endif 
