




#ifndef BOOST_GEOMETRY_VIEWS_CLOSEABLE_VIEW_HPP
#define BOOST_GEOMETRY_VIEWS_CLOSEABLE_VIEW_HPP

#include <boost/geometry/core/closure.hpp>
#include <boost/geometry/core/ring_type.hpp>
#include <boost/geometry/core/tag.hpp>
#include <boost/geometry/core/tags.hpp>
#include <boost/geometry/iterators/closing_iterator.hpp>

#include <boost/geometry/views/identity_view.hpp>

namespace boost { namespace geometry
{

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4512)
#endif

#ifndef DOXYGEN_NO_DETAIL

namespace detail
{

template <typename Range>
struct closing_view
{
explicit inline closing_view(Range& r)
: m_range(r)
{}

typedef closing_iterator<Range> iterator;
typedef closing_iterator<Range const> const_iterator;

inline const_iterator begin() const { return const_iterator(m_range); }
inline const_iterator end() const { return const_iterator(m_range, true); }

inline iterator begin() { return iterator(m_range); }
inline iterator end() { return iterator(m_range, true); }
private :
Range& m_range;
};

}

#endif 



template <typename Range, closure_selector Close>
struct closeable_view {};


#ifndef DOXYGEN_NO_SPECIALIZATIONS

template <typename Range>
struct closeable_view<Range, closed>
{
typedef identity_view<Range> type;
};


template <typename Range>
struct closeable_view<Range, open>
{
typedef detail::closing_view<Range> type;
};

#endif 


#if defined(_MSC_VER)
#pragma warning(pop)
#endif

}} 


#endif 
