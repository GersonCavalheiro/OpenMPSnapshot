


#ifndef BOOST_IOSTREAMS_DETAIL_DOUBLE_OBJECT_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_DOUBLE_OBJECT_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <algorithm>              
#include <boost/detail/workaround.hpp>
#include <boost/mpl/if.hpp>
#if BOOST_WORKAROUND(__MWERKS__, > 0x3003)
# include <msl_utility>
#else
# include <boost/call_traits.hpp>
#endif

namespace boost { namespace iostreams { namespace detail {

template<typename T>
class single_object_holder {
public:
#if BOOST_WORKAROUND(__MWERKS__, > 0x3003)
typedef Metrowerks::call_traits<T>             traits_type;
#else
typedef boost::call_traits<T>                  traits_type;
#endif
typedef typename traits_type::param_type       param_type;
typedef typename traits_type::reference        reference;
typedef typename traits_type::const_reference  const_reference;
single_object_holder() { }
single_object_holder(param_type t) : first_(t) { }
reference first() { return first_; }
const_reference first() const { return first_; }
reference second() { return first_; }
const_reference second() const { return first_; }
void swap(single_object_holder& o)
{ std::swap(first_, o.first_); }
private:
T first_;
};

template<typename T>
struct double_object_holder {
public:
#if BOOST_WORKAROUND(__MWERKS__, > 0x3003)
typedef Metrowerks::call_traits<T>             traits_type;
#else
typedef boost::call_traits<T>                  traits_type;
#endif
typedef typename traits_type::param_type       param_type;
typedef typename traits_type::reference        reference;
typedef typename traits_type::const_reference  const_reference;
double_object_holder() { }
double_object_holder(param_type t1, param_type t2)
: first_(t1), second_(t2) { }
reference first() { return first_; }
const_reference first() const { return first_; }
reference second() { return second_; }
const_reference second() const { return second_; }
void swap(double_object_holder& d)
{ 
std::swap(first_, d.first_); 
std::swap(second_, d.second_); 
}
private:
T first_, second_;
};

template<typename T, typename IsDouble>
class double_object 
: public mpl::if_<
IsDouble, 
double_object_holder<T>, 
single_object_holder<T>
>::type
{
private:
typedef typename 
mpl::if_<
IsDouble, 
double_object_holder<T>, 
single_object_holder<T>
>::type                                base_type;
public:
#if BOOST_WORKAROUND(__MWERKS__, > 0x3003)
typedef Metrowerks::call_traits<T>             traits_type;
#else
typedef boost::call_traits<T>                  traits_type;
#endif
typedef typename traits_type::param_type       param_type;
typedef typename traits_type::reference        reference;
typedef typename traits_type::const_reference  const_reference;
double_object() : base_type() {}
double_object(param_type t1, param_type t2)
: base_type(t1, t2) { }
bool is_double() const { return IsDouble::value; }
};

} } } 

#endif 
