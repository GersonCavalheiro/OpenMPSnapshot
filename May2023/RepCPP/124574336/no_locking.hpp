

#ifndef BOOST_FLYWEIGHT_NO_LOCKING_HPP
#define BOOST_FLYWEIGHT_NO_LOCKING_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/no_locking_fwd.hpp>
#include <boost/flyweight/locking_tag.hpp>



namespace boost{

namespace flyweights{

struct no_locking:locking_marker
{
struct             mutex_type{};
typedef mutex_type lock_type;
};

} 

} 

#endif
