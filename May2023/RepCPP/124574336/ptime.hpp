



#ifndef BOOST_ICL_PTIME_HPP_JOFA_080416
#define BOOST_ICL_PTIME_HPP_JOFA_080416

#include <boost/icl/detail/boost_config.hpp>
#include <boost/detail/workaround.hpp>

#ifdef BOOST_MSVC 
#pragma warning(push)
#pragma warning(disable:4100) 
#pragma warning(disable:4127) 
#pragma warning(disable:4244) 
#pragma warning(disable:4702) 
#pragma warning(disable:4996) 
#endif                        

#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include <boost/date_time/posix_time/posix_time.hpp>

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/icl/type_traits/identity_element.hpp>
#include <boost/icl/type_traits/difference_type_of.hpp>
#include <boost/icl/type_traits/size_type_of.hpp>
#include <boost/icl/type_traits/is_discrete.hpp>

namespace boost{namespace icl
{
template<> struct is_discrete<boost::posix_time::ptime>
{
typedef is_discrete type;
BOOST_STATIC_CONSTANT(bool, value = true);
};

template<> 
inline boost::posix_time::ptime identity_element<boost::posix_time::ptime>::value()
{ 
return boost::posix_time::ptime(boost::posix_time::min_date_time); 
}

template<> 
struct has_difference<boost::posix_time::ptime> 
{ 
typedef has_difference type;
BOOST_STATIC_CONSTANT(bool, value = true);
};  

template<> 
struct difference_type_of<boost::posix_time::ptime> 
{ 
typedef boost::posix_time::time_duration type; 
};  

template<> 
struct size_type_of<boost::posix_time::ptime> 
{ 
typedef boost::posix_time::time_duration type; 
};  

inline boost::posix_time::ptime operator ++(boost::posix_time::ptime& x)
{
return x += boost::posix_time::ptime::time_duration_type::unit();
}

inline boost::posix_time::ptime operator --(boost::posix_time::ptime& x)
{
return x -= boost::posix_time::ptime::time_duration_type::unit();
}

template<> struct is_discrete<boost::posix_time::time_duration>
{
typedef is_discrete type;
BOOST_STATIC_CONSTANT(bool, value = true);
};

template<> 
struct has_difference<boost::posix_time::time_duration> 
{ 
typedef has_difference type;
BOOST_STATIC_CONSTANT(bool, value = true);
};  

template<> 
struct size_type_of<boost::posix_time::time_duration> 
{ 
typedef boost::posix_time::time_duration type; 
};  

inline boost::posix_time::time_duration operator ++(boost::posix_time::time_duration& x)
{
return x += boost::posix_time::ptime::time_duration_type::unit();
}

inline boost::posix_time::time_duration operator --(boost::posix_time::time_duration& x)
{
return x -= boost::posix_time::ptime::time_duration_type::unit();
}
}} 

#endif


