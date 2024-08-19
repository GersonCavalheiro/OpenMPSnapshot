#ifndef BOOST_MOVE_DETAIL_IS_SORTED_HPP
#define BOOST_MOVE_DETAIL_IS_SORTED_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace movelib {

template<class ForwardIt, class Pred>
bool is_sorted(ForwardIt const first, ForwardIt last, Pred pred)
{
if (first != last) {
ForwardIt next = first, cur(first);
while (++next != last) {
if (pred(*next, *cur))
return false;
cur = next;
}
}
return true;
}

template<class ForwardIt, class Pred>
bool is_sorted_and_unique(ForwardIt first, ForwardIt last, Pred pred)
{
if (first != last) {
ForwardIt next = first;
while (++next != last) {
if (!pred(*first, *next))
return false;
first = next;
}
}
return true;
}

}  
}  

#endif   
