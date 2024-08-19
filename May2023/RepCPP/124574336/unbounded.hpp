

#ifndef BOOST_MULTI_INDEX_DETAIL_UNBOUNDED_HPP
#define BOOST_MULTI_INDEX_DETAIL_UNBOUNDED_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>

namespace boost{

namespace multi_index{





namespace detail{class unbounded_helper;}

detail::unbounded_helper unbounded(detail::unbounded_helper);

namespace detail{

class unbounded_helper
{
unbounded_helper(){}
unbounded_helper(const unbounded_helper&){}
friend unbounded_helper multi_index::unbounded(unbounded_helper);
};

typedef unbounded_helper (*unbounded_type)(unbounded_helper);

} 

inline detail::unbounded_helper unbounded(detail::unbounded_helper)
{
return detail::unbounded_helper();
}



namespace detail{

struct none_unbounded_tag{};
struct lower_unbounded_tag{};
struct upper_unbounded_tag{};
struct both_unbounded_tag{};

} 

} 

} 

#endif
