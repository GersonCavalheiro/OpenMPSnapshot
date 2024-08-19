




#ifndef BOOST_GEOMETRY_VIEWS_IDENTITY_VIEW_HPP
#define BOOST_GEOMETRY_VIEWS_IDENTITY_VIEW_HPP


#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>


namespace boost { namespace geometry
{

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4512)
#endif


template <typename Range>
struct identity_view
{
typedef typename boost::range_iterator<Range const>::type const_iterator;
typedef typename boost::range_iterator<Range>::type iterator;

explicit inline identity_view(Range& r)
: m_range(r)
{}

inline const_iterator begin() const { return boost::begin(m_range); }
inline const_iterator end() const { return boost::end(m_range); }

inline iterator begin() { return boost::begin(m_range); }
inline iterator end() { return boost::end(m_range); }
private :
Range& m_range;
};

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}} 


#endif 
