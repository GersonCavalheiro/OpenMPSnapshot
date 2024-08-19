#ifndef BOOST_BAD_OPTIONAL_ACCESS_22MAY2014_HPP
#define BOOST_BAD_OPTIONAL_ACCESS_22MAY2014_HPP

#include <stdexcept>
#if __cplusplus < 201103L
#include <string> 
#endif

namespace boost {

#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wweak-vtables"
#endif

class bad_optional_access : public std::logic_error
{
public:
bad_optional_access()
: std::logic_error("Attempted to access the value of an uninitialized optional object.")
{}
};

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

} 

#endif
