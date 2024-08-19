#ifndef BOOST_SERIALIZATION_SINGLETON_HPP
#define BOOST_SERIALIZATION_SINGLETON_HPP


#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/serialization/force_include.hpp>
#include <boost/serialization/config.hpp>

#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4511 4512)
#endif

namespace boost {
namespace serialization {









class BOOST_SYMBOL_VISIBLE singleton_module :
public boost::noncopyable
{
private:
BOOST_DLLEXPORT bool & get_lock() BOOST_USED {
static bool lock = false;
return lock;
}

public:
BOOST_DLLEXPORT void lock(){
get_lock() = true;
}
BOOST_DLLEXPORT void unlock(){
get_lock() = false;
}
BOOST_DLLEXPORT bool is_locked(){
return get_lock();
}
};

static inline singleton_module & get_singleton_module(){
static singleton_module m;
return m;
}

namespace detail {

template<class T>
class singleton_wrapper : public T
{
static bool & get_is_destroyed(){
static bool is_destroyed_flag = false;
return is_destroyed_flag;
}
public:
singleton_wrapper(){
BOOST_ASSERT(! is_destroyed());
}
~singleton_wrapper(){
get_is_destroyed() = true;
}
static bool is_destroyed(){
return get_is_destroyed();
}
};

} 

template <class T>
class singleton {
private:
static T * m_instance;
static void use(T const &) {}
static T & get_instance() {
BOOST_ASSERT(! is_destroyed());

static detail::singleton_wrapper< T > t;

if (m_instance) use(* m_instance);

return static_cast<T &>(t);
}
protected:
BOOST_DLLEXPORT singleton(){}

public:
BOOST_DLLEXPORT static T & get_mutable_instance(){
BOOST_ASSERT(! get_singleton_module().is_locked());
return get_instance();
}
BOOST_DLLEXPORT static const T & get_const_instance(){
return get_instance();
}
BOOST_DLLEXPORT static bool is_destroyed(){
return detail::singleton_wrapper< T >::is_destroyed();
}
};

template<class T>
T * singleton< T >::m_instance = & singleton< T >::get_instance();

} 
} 

#include <boost/archive/detail/abi_suffix.hpp> 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif 
