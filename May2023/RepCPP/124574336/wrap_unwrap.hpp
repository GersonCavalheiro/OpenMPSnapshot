

#ifndef BOOST_IOSTREAMS_DETAIL_WRAP_UNWRAP_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_WRAP_UNWRAP_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#include <boost/config.hpp>                             
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/enable_if_stream.hpp>
#include <boost/iostreams/traits_fwd.hpp>               
#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/if.hpp>
#include <boost/ref.hpp>

namespace boost { namespace iostreams { namespace detail {


template<typename T>
struct wrapped_type 
: mpl::if_<is_std_io<T>, reference_wrapper<T>, T>
{ };

template<typename T>
struct unwrapped_type 
: unwrap_reference<T>
{ };

template<typename T>
struct unwrap_ios 
: mpl::eval_if<
is_std_io<T>, 
unwrap_reference<T>, 
mpl::identity<T>
>
{ };


#ifndef BOOST_NO_SFINAE 
template<typename T>
inline T wrap(const T& t BOOST_IOSTREAMS_DISABLE_IF_STREAM(T)) 
{ return t; }

template<typename T>
inline typename wrapped_type<T>::type
wrap(T& t BOOST_IOSTREAMS_ENABLE_IF_STREAM(T)) { return boost::ref(t); }
#else 
template<typename T>
inline typename wrapped_type<T>::type 
wrap_impl(const T& t, mpl::true_) { return boost::ref(const_cast<T&>(t)); }

template<typename T>
inline typename wrapped_type<T>::type 
wrap_impl(T& t, mpl::true_) { return boost::ref(t); }

template<typename T>
inline typename wrapped_type<T>::type 
wrap_impl(const T& t, mpl::false_) { return t; }

template<typename T>
inline typename wrapped_type<T>::type 
wrap_impl(T& t, mpl::false_) { return t; }

template<typename T>
inline typename wrapped_type<T>::type 
wrap(const T& t) { return wrap_impl(t, is_std_io<T>()); }

template<typename T>
inline typename wrapped_type<T>::type 
wrap(T& t) { return wrap_impl(t, is_std_io<T>()); }
#endif 


template<typename T>
typename unwrapped_type<T>::type& 
unwrap(const reference_wrapper<T>& ref) { return ref.get(); }

template<typename T>
typename unwrapped_type<T>::type& unwrap(T& t) { return t; }

template<typename T>
const typename unwrapped_type<T>::type& unwrap(const T& t) { return t; }

} } } 

#endif 
