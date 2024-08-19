

#ifndef BOOST_MULTI_INDEX_DETAIL_HEADER_HOLDER_HPP
#define BOOST_MULTI_INDEX_DETAIL_HEADER_HOLDER_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/noncopyable.hpp>

namespace boost{

namespace multi_index{

namespace detail{



template<typename NodeTypePtr,typename Final>
struct header_holder:private noncopyable
{
header_holder():member(final().allocate_node()){}
~header_holder(){final().deallocate_node(&*member);}

NodeTypePtr member;

private:
Final& final(){return *static_cast<Final*>(this);}
};

} 

} 

} 

#endif
