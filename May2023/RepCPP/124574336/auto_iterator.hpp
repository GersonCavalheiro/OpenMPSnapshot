

#ifndef BOOST_POLY_COLLECTION_DETAIL_AUTO_ITERATOR_HPP
#define BOOST_POLY_COLLECTION_DETAIL_AUTO_ITERATOR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/iterator/iterator_adaptor.hpp>

namespace boost{

namespace poly_collection{

namespace detail{



template<typename Iterator>
class auto_iterator:
public boost::iterator_adaptor<auto_iterator<Iterator>,Iterator,Iterator>
{
public:
auto_iterator()=default;
auto_iterator(const Iterator& it):auto_iterator::iterator_adaptor_{it}{}
auto_iterator(const auto_iterator&)=default;
auto_iterator& operator=(const auto_iterator&)=default;

private:
friend class boost::iterator_core_access;

Iterator& dereference()const noexcept
{
return const_cast<auto_iterator*>(this)->base_reference();
}
};

} 

} 

} 

#endif
