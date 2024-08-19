

#ifndef BOOST_FLYWEIGHT_NO_TRACKING_HPP
#define BOOST_FLYWEIGHT_NO_TRACKING_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/no_tracking_fwd.hpp>
#include <boost/flyweight/tracking_tag.hpp>



namespace boost{

namespace flyweights{

struct no_tracking:tracking_marker
{
struct entry_type
{
template<typename Value,typename Key>
struct apply{typedef Value type;};
};

struct handle_type
{
template<typename Handle,typename TrackingHelper>
struct apply{typedef Handle type;};
};
};

} 

} 

#endif
