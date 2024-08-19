

#ifndef BOOST_FLYWEIGHT_SERIALIZE_HPP
#define BOOST_FLYWEIGHT_SERIALIZE_HPP

#if defined(_MSC_VER)&&(_MSC_VER>=1200)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/flyweight/flyweight_fwd.hpp>
#include <boost/flyweight/detail/archive_constructed.hpp>
#include <boost/flyweight/detail/serialization_helper.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/throw_exception.hpp> 
#include <memory>



namespace boost{

namespace serialization{

template<
class Archive,
typename T,typename Arg1,typename Arg2,typename Arg3
>
inline void serialize(
Archive& ar,::boost::flyweights::flyweight<T,Arg1,Arg2,Arg3>& f,
const unsigned int version)
{
split_free(ar,f,version);              
}                                               

template<
class Archive,
typename T,typename Arg1,typename Arg2,typename Arg3
>
void save(
Archive& ar,const ::boost::flyweights::flyweight<T,Arg1,Arg2,Arg3>& f,
const unsigned int )
{
typedef ::boost::flyweights::flyweight<T,Arg1,Arg2,Arg3>    flyweight;
typedef ::boost::flyweights::detail::save_helper<flyweight> helper;
typedef typename helper::size_type                          size_type;

helper& hlp=ar.template get_helper<helper>();

size_type n=hlp.find(f);
ar<<make_nvp("item",n);
if(n==hlp.size()){
ar<<make_nvp("key",f.get_key());
hlp.push_back(f);
}
}

template<
class Archive,
typename T,typename Arg1,typename Arg2,typename Arg3
>
void load(
Archive& ar,::boost::flyweights::flyweight<T,Arg1,Arg2,Arg3>& f,
const unsigned int version)
{
typedef ::boost::flyweights::flyweight<T,Arg1,Arg2,Arg3>    flyweight;
typedef typename flyweight::key_type                        key_type;
typedef ::boost::flyweights::detail::load_helper<flyweight> helper;
typedef typename helper::size_type                          size_type;

helper& hlp=ar.template get_helper<helper>();

size_type n=0;
ar>>make_nvp("item",n);
if(n>hlp.size()){
throw_exception(
archive::archive_exception(archive::archive_exception::other_exception));
}
else if(n==hlp.size()){
::boost::flyweights::detail::archive_constructed<key_type> k(
"key",ar,version);
hlp.push_back(flyweight(k.get()));
}
f=hlp[n];
}

} 

} 

#endif
