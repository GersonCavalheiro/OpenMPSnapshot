

#ifndef BOOST_MULTI_INDEX_DETAIL_CONVERTER_HPP
#define BOOST_MULTI_INDEX_DETAIL_CONVERTER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost{

namespace multi_index{

namespace detail{



template<typename MultiIndexContainer,typename Index>
struct converter
{
static const Index& index(const MultiIndexContainer& x){return x;}
static Index&       index(MultiIndexContainer& x){return x;}

static typename Index::const_iterator const_iterator(
const MultiIndexContainer& x,
typename MultiIndexContainer::final_node_type* node)
{
return x.Index::make_iterator(node);
}

static typename Index::iterator iterator(
MultiIndexContainer& x,
typename MultiIndexContainer::final_node_type* node)
{
return x.Index::make_iterator(node);
}
};

} 

} 

} 

#endif
