#ifndef BOOST_SERIALIZATION_TRACKING_ENUM_HPP
#define BOOST_SERIALIZATION_TRACKING_ENUM_HPP

#if defined(_MSC_VER)
# pragma once
#endif




namespace boost {
namespace serialization {


enum tracking_type
{
track_never = 0,
track_selectively = 1,
track_always = 2
};

} 
} 

#endif 
