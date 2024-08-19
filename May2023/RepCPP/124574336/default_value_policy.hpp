

#ifndef BOOST_FLYWEIGHT_DETAIL_DEFAULT_VALUE_POLICY_HPP
#define BOOST_FLYWEIGHT_DETAIL_DEFAULT_VALUE_POLICY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/flyweight/detail/perfect_fwd.hpp>
#include <boost/flyweight/detail/value_tag.hpp>



namespace boost{

namespace flyweights{

namespace detail{

template<typename Value>
struct default_value_policy:value_marker
{
typedef Value key_type;
typedef Value value_type;

struct rep_type
{


#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)&&\
!defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)&&\
BOOST_WORKAROUND(BOOST_GCC,<=40603)


rep_type():x(){}
#endif

#define BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY(args) \
:x(BOOST_FLYWEIGHT_FORWARD(args)){}

BOOST_FLYWEIGHT_PERFECT_FWD(
explicit rep_type,
BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY)

#undef BOOST_FLYWEIGHT_PERFECT_FWD_CTR_BODY

rep_type(const rep_type& r):x(r.x){}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
rep_type(rep_type&& r):x(std::move(r.x)){}
#endif

operator const value_type&()const{return x;}

value_type x;
};

static void construct_value(const rep_type&){}
static void copy_value(const rep_type&){}

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
static void move_value(const rep_type&){}
#endif
};

} 

} 

} 

#endif
