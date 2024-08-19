
#ifndef BOOST_ICL_GREGORIAN_DATE_HPP_JOFA_080416
#define BOOST_ICL_GREGORIAN_DATE_HPP_JOFA_080416

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
#include <boost/date_time/gregorian/gregorian.hpp>

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#include <boost/icl/type_traits/identity_element.hpp>
#include <boost/icl/type_traits/is_discrete.hpp>
#include <boost/icl/type_traits/difference_type_of.hpp>
#include <boost/icl/type_traits/size_type_of.hpp>

namespace boost{namespace icl
{
template<> struct is_discrete<boost::gregorian::date>
{
typedef is_discrete type;
BOOST_STATIC_CONSTANT(bool, value = true);
};

template<> 
inline boost::gregorian::date identity_element<boost::gregorian::date>::value()
{ 
return boost::gregorian::date(boost::gregorian::min_date_time); 
}

template<> 
struct identity_element<boost::gregorian::date_duration>
{
static boost::gregorian::date_duration value()
{ 
return boost::gregorian::date(boost::gregorian::min_date_time) 
- boost::gregorian::date(boost::gregorian::min_date_time); 
}
};

template<> 
struct has_difference<boost::gregorian::date> 
{ 
typedef has_difference type;
BOOST_STATIC_CONSTANT(bool, value = true);
};  

template<> 
struct difference_type_of<boost::gregorian::date> 
{ typedef boost::gregorian::date_duration type; };  

template<> 
struct size_type_of<boost::gregorian::date> 
{ typedef boost::gregorian::date_duration type; };  



inline boost::gregorian::date operator ++(boost::gregorian::date& x)
{
return x += boost::gregorian::date::duration_type::unit();
}

inline boost::gregorian::date operator --(boost::gregorian::date& x)
{
return x -= boost::gregorian::date::duration_type::unit();
}

template<> struct is_discrete<boost::gregorian::date_duration>
{
typedef is_discrete type;
BOOST_STATIC_CONSTANT(bool, value = true);
};

template<> 
struct has_difference<boost::gregorian::date_duration> 
{ 
typedef has_difference type;
BOOST_STATIC_CONSTANT(bool, value = true);
};  

template<> 
struct size_type_of<boost::gregorian::date_duration> 
{ 
typedef boost::gregorian::date_duration type; 
};  

inline boost::gregorian::date_duration operator ++(boost::gregorian::date_duration& x)
{
return x += boost::gregorian::date::duration_type::unit();
}

inline boost::gregorian::date_duration operator --(boost::gregorian::date_duration& x)
{
return x -= boost::gregorian::date::duration_type::unit();
}



}} 

#endif


