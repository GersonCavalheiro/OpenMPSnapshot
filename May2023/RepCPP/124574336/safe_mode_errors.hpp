

#ifndef BOOST_MULTI_INDEX_SAFE_MODE_ERRORS_HPP
#define BOOST_MULTI_INDEX_SAFE_MODE_ERRORS_HPP

#if defined(_MSC_VER)
#pragma once
#endif

namespace boost{

namespace multi_index{

namespace safe_mode{



enum error_code
{
invalid_iterator=0,
not_dereferenceable_iterator,
not_incrementable_iterator,
not_decrementable_iterator,
not_owner,
not_same_owner,
invalid_range,
inside_range,
out_of_bounds,
same_container,
unequal_allocators
};

} 

} 

} 

#endif
